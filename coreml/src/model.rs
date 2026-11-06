//! The compiled multifunction model and its predictions.

use std::cell::RefCell;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use block2::StackBlock;
use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLArrayBatchProvider, MLBatchProvider, MLComputeUnits, MLDictionaryFeatureProvider,
    MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration, MLMultiArray,
    MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSInteger, NSNumber, NSString, NSURL};

use crate::{DataType, Element, Error, Input, Manifest, Tensor};

/// The hardware a [`Model`]'s functions run on: Core ML's
/// `MLComputeUnits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeUnits {
    /// The CPU alone.
    CpuOnly,
    /// The GPU, with the CPU for what the GPU does not run.
    CpuAndGpu,
    /// The Neural Engine, with the CPU for what it does not run.
    CpuAndNeuralEngine,
    /// Whatever Core ML picks among CPU, GPU and Neural Engine.
    All,
}

impl ComputeUnits {
    fn ml(self) -> MLComputeUnits {
        match self {
            Self::CpuOnly => MLComputeUnits::CPUOnly,
            Self::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            Self::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
            Self::All => MLComputeUnits::All,
        }
    }
}

/// One function's `MLModel`.
struct Bucket {
    /// The sequence length the function accepts.
    len: usize,
    model: Retained<MLModel>,
}

/// A compiled multifunction Core ML model: one function per sequence-length
/// bucket, all loaded from the same `.mlmodelc` so they share its weights.
///
/// A prediction pads its inputs to the smallest bucket that holds them and
/// runs that function. Prediction is blocking; the caller owns the thread
/// and its QoS.
pub struct Model {
    manifest: Manifest,
    /// One per manifest function, in the same (ascending) order.
    buckets: Box<[Bucket]>,
}

// SAFETY: `MLModel` may be used from any thread provided calls on the same
// object do not overlap. `Model` is `!Sync` and every method that touches
// its `MLModel`s takes `&mut self`, so those calls never overlap; moving the
// model between threads preserves that.
unsafe impl Send for Model {}

impl Model {
    /// Loads a model directory — `manifest.json` beside exactly one
    /// compiled `*.mlmodelc` — with every function on `units`.
    ///
    /// # Errors
    ///
    /// The [`Manifest::load`] failures, [`Error::ModelDir`] without exactly
    /// one bundle, [`Error::CoreMl`] when Core ML refuses a function.
    pub fn open(dir: impl AsRef<Path>, units: ComputeUnits) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let manifest = Manifest::load(dir)?;
        let bundle = compiled_bundle(dir)?;
        let url = NSURL::fileURLWithPath_isDirectory(
            &NSString::from_str(&bundle.to_string_lossy()),
            true,
        );
        let buckets = manifest
            .functions()
            .iter()
            .map(|function| {
                // SAFETY: plain Objective-C allocation and init; there are no
                // invariants the caller must uphold.
                let config = unsafe { MLModelConfiguration::new() };
                // SAFETY: `config` is a live configuration, `units.ml()` a
                // valid `MLComputeUnits`, and `function.name` names a
                // function the manifest declares.
                unsafe {
                    config.setComputeUnits(units.ml());
                    config.setFunctionName(Some(&NSString::from_str(&function.name)));
                }
                // SAFETY: `url` is the compiled bundle and `config` a valid
                // configuration; a load failure is reported through `error`.
                let model =
                    unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
                        .map_err(core_ml)?;
                Ok(Bucket {
                    len: function.len,
                    model,
                })
            })
            .collect::<Result<Box<[Bucket]>, Error>>()?;
        tracing::debug!(
            bundle = %bundle.display(),
            functions = buckets.len(),
            ?units,
            "Core ML model loaded"
        );
        Ok(Self { manifest, buckets })
    }

    /// The model's manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Runs one prediction: `inputs`, all of one length, padded to the
    /// smallest bucket that holds them.
    ///
    /// # Panics
    ///
    /// When `inputs` is empty or its inputs differ in length: one
    /// prediction is one sequence.
    ///
    /// # Errors
    ///
    /// [`Error::TooLong`] past the largest bucket, [`Error::UnknownInput`]
    /// for an input the manifest does not declare, and [`Error::CoreMl`].
    pub fn predict(&mut self, inputs: &[&dyn Input]) -> Result<Prediction, Error> {
        let bucket = self.bucket_for(sequence_len(inputs))?;
        let provider = features(&self.manifest, inputs, bucket.len)?;
        // SAFETY: `provider` is a valid MLFeatureProvider.
        let features = unsafe {
            bucket
                .model
                .predictionFromFeatures_error(&ProtocolObject::from_retained(provider))
        }
        .map_err(core_ml)?;
        Ok(Prediction { features })
    }

    /// Runs one Core ML batch prediction: each item a [`predict`]-style
    /// input set, all padded to the smallest bucket that holds the longest.
    ///
    /// [`predict`]: Self::predict
    ///
    /// # Panics
    ///
    /// When an item is empty or its inputs differ in length, and when Core
    /// ML breaks its contract by returning another number of predictions
    /// than items.
    ///
    /// # Errors
    ///
    /// As [`predict`](Self::predict).
    pub fn predict_batch(&mut self, batch: &[&[&dyn Input]]) -> Result<Vec<Prediction>, Error> {
        let Some(longest) = batch.iter().map(|inputs| sequence_len(inputs)).max() else {
            return Ok(Vec::new());
        };
        let bucket = self.bucket_for(longest)?;
        let providers = batch
            .iter()
            .map(|inputs| {
                features(&self.manifest, inputs, bucket.len).map(ProtocolObject::from_retained)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let providers = NSArray::from_retained_slice(&providers);
        // SAFETY: `providers` holds valid MLFeatureProviders.
        let provider = unsafe {
            MLArrayBatchProvider::initWithFeatureProviderArray(
                MLArrayBatchProvider::alloc(),
                &providers,
            )
        };
        // SAFETY: `provider` is a valid MLBatchProvider.
        let out = unsafe {
            bucket
                .model
                .predictionsFromBatch_error(&ProtocolObject::from_retained(provider))
        }
        .map_err(core_ml)?;
        // SAFETY: `out` is a batch provider the model returned.
        let count = unsafe { out.count() } as usize;
        assert_eq!(
            count,
            batch.len(),
            "Core ML's predictionsFromBatch returned {count} predictions for {} items",
            batch.len()
        );
        Ok((0..count)
            .map(|i| Prediction {
                // SAFETY: `i` is below `out.count()`.
                features: unsafe { out.featuresAtIndex(i as NSInteger) },
            })
            .collect())
    }

    /// The smallest bucket that holds `len`.
    fn bucket_for(&self, len: usize) -> Result<&Bucket, Error> {
        self.buckets
            .iter()
            .find(|bucket| len <= bucket.len)
            .ok_or(Error::TooLong {
                len,
                max: self.manifest.max_len(),
            })
    }
}

/// The outputs of one prediction.
pub struct Prediction {
    features: Retained<ProtocolObject<dyn MLFeatureProvider>>,
}

impl Prediction {
    /// The output `name`, copied out as a [`Tensor`] of `T`.
    ///
    /// # Errors
    ///
    /// [`Error::MissingOutput`] when the prediction has no multi-array
    /// `name`, [`Error::OutputType`] when its elements are not `T`.
    pub fn output<T: Element>(&self, name: &str) -> Result<Tensor<T>, Error> {
        let missing = || Error::MissingOutput(name.to_owned());
        // SAFETY: `features` is a feature provider the model returned.
        let value = unsafe { self.features.featureValueForName(&NSString::from_str(name)) }
            .ok_or_else(missing)?;
        // SAFETY: `value` is a live feature value; `multiArrayValue` returns
        // nil when the feature is not a multi-array.
        let array = unsafe { value.multiArrayValue() }.ok_or_else(missing)?;
        // SAFETY: `array` is a live multi-array.
        let actual = unsafe { array.dataType() };
        if actual != ml_data_type(T::DATA_TYPE) {
            return Err(Error::OutputType {
                name: name.to_owned(),
                expected: T::DATA_TYPE,
                actual,
            });
        }
        // SAFETY: `array` is a live multi-array.
        let shape: Vec<usize> = unsafe { array.shape() }
            .iter()
            .map(|n| n.as_usize())
            .collect();
        // SAFETY: `array` is a live multi-array.
        let strides: Vec<isize> = unsafe { array.strides() }
            .iter()
            .map(|n| n.as_isize())
            .collect();
        let bytes = RefCell::new(Vec::new());
        let copy = StackBlock::new(|ptr: NonNull<c_void>, size: NSInteger| {
            // SAFETY: `ptr` is the array's storage, `size` bytes, valid for
            // the block's duration; the copy reads exactly those bytes and
            // keeps no pointer past the block.
            let storage =
                unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>(), size as usize) };
            bytes.borrow_mut().extend_from_slice(storage);
        });
        // SAFETY: `copy` reads only within `ptr..ptr + size` while it runs.
        unsafe { array.getBytesWithHandler(&copy) };
        Ok(Tensor::gather(&bytes.into_inner(), &shape, &strides))
    }
}

/// The sequence length `inputs` share.
fn sequence_len(inputs: &[&dyn Input]) -> usize {
    let len = inputs.first().expect("a prediction has inputs").len();
    assert!(
        inputs.iter().all(|input| input.len() == len),
        "the inputs of one prediction differ in length"
    );
    len
}

/// The feature provider for `inputs` padded to `len`.
fn features(
    manifest: &Manifest,
    inputs: &[&dyn Input],
    len: usize,
) -> Result<Retained<MLDictionaryFeatureProvider>, Error> {
    let mut names = Vec::with_capacity(inputs.len());
    let mut values = Vec::with_capacity(inputs.len());
    for input in inputs {
        let spec = manifest
            .inputs()
            .get(input.name())
            .ok_or_else(|| Error::UnknownInput(input.name().to_owned()))?;
        let rank = spec.shape.len();
        let shape: Vec<Retained<NSNumber>> = (0..rank)
            .map(|axis| NSNumber::new_usize(if axis + 1 == rank { len } else { 1 }))
            .collect();
        let shape = NSArray::from_retained_slice(&shape);
        // SAFETY: `alloc` and `initWithShape:dataType:` are sound here; the
        // returned array owns its elements and stays alive through the
        // feature provider, so Core ML's deferred release frees memory it
        // owns.
        let array = unsafe {
            MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &shape,
                ml_data_type(input.data_type()),
            )
        }
        .map_err(core_ml)?;
        // ObjC handler blocks must not panic: the block only reads the last
        // stride and hands it to `fill`, which writes in bounds.
        let fill = StackBlock::new(
            |ptr: NonNull<c_void>, _size: NSInteger, strides: NonNull<NSArray<NSNumber>>| {
                // SAFETY: `strides` is a live `NSArray` holding one entry per
                // dimension, `rank` of them, so index `rank - 1` is in bounds.
                let stride = unsafe { strides.as_ref() }
                    .objectAtIndex(rank - 1)
                    .as_usize();
                // SAFETY: `ptr` is the storage of a `[1, …, 1, len]` array of
                // `input.data_type()`, so it holds `len` elements `stride`
                // apart along the last axis, writable for the block's
                // duration; `len >= input.len()` since the bucket holds it.
                unsafe { input.fill(ptr, stride, len) };
            },
        );
        // SAFETY: `fill` writes only within the elements `ptr` and `strides`
        // describe, and keeps no pointer past the block.
        unsafe { array.getMutableBytesWithHandler(&fill) };
        names.push(NSString::from_str(input.name()));
        // SAFETY: `array` is a live multi-array.
        values.push(unsafe { MLFeatureValue::featureValueWithMultiArray(&array) });
    }
    let names: Vec<&NSString> = names.iter().map(|name| &**name).collect();
    let values: Vec<&AnyObject> = values.iter().map(|value| value.as_ref()).collect();
    let features: Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_slices(&names, &values);
    // SAFETY: `features` pairs NSString keys with feature values.
    unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &features,
        )
    }
    .map_err(core_ml)
}

fn ml_data_type(data_type: DataType) -> MLMultiArrayDataType {
    match data_type {
        DataType::Int32 => MLMultiArrayDataType::Int32,
        DataType::Float16 => MLMultiArrayDataType::Float16,
        DataType::Float32 => MLMultiArrayDataType::Float32,
    }
}

/// The one `*.mlmodelc` bundle in `dir`, canonicalized.
fn compiled_bundle(dir: &Path) -> Result<PathBuf, Error> {
    let io = |source| Error::Io {
        path: dir.to_path_buf(),
        source,
    };
    let mut bundles = std::fs::read_dir(dir)
        .map_err(io)?
        .map(|entry| entry.map(|entry| entry.path()).map_err(io))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|path| path.extension().is_some_and(|e| e == "mlmodelc"));
    let bundle = bundles
        .next()
        .ok_or_else(|| Error::ModelDir(dir.to_path_buf()))?;
    if bundles.next().is_some() {
        return Err(Error::ModelDir(dir.to_path_buf()));
    }
    bundle.canonicalize().map_err(|source| Error::Io {
        path: bundle,
        source,
    })
}

fn core_ml(error: Retained<NSError>) -> Error {
    Error::CoreMl {
        domain: error.domain().to_string(),
        code: error.code(),
        message: error.localizedDescription().to_string(),
    }
}
