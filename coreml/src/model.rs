//! Compiled artifacts, independently loaded functions, and owned stateful sessions.
use crate::{ComputeUnits, DataType, Element, Error, Inputs, Tensor};
use objc2::{
    AllocAnyThread,
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_core_ml::{
    MLArrayBatchProvider, MLBatchProvider, MLFeatureDescription, MLFeatureProvider, MLModel,
    MLModelConfiguration, MLState,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSInteger, NSString, NSURL};
use std::{
    cell::Cell,
    collections::BTreeMap,
    marker::PhantomData,
    path::{Path, PathBuf},
};

/// A tensor feature's native default shape. Flexible constraints are enforced
/// by Core ML; this shape is descriptive and is not a padding policy.
#[derive(Debug, Clone)]
pub struct FeatureDescription {
    pub data_type: DataType,
    pub default_shape: Vec<usize>,
    pub optional: bool,
}
/// A loaded function's features, read from the compiled model itself.
#[derive(Debug, Clone)]
pub struct FunctionDescription {
    pub inputs: BTreeMap<String, FeatureDescription>,
    pub outputs: BTreeMap<String, FeatureDescription>,
    pub states: BTreeMap<String, FeatureDescription>,
}
/// A compiled model artifact. Functions are loaded explicitly and may have
/// different input/output contracts. No tokenizer or bucketing policy is implied.
#[derive(Debug, Clone)]
pub struct Model {
    path: PathBuf,
}
impl Model {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let path = path.canonicalize().map_err(|source| Error::Io {
            path: path.into(),
            source,
        })?;
        Ok(Self { path })
    }
    /// Load a named function, or the model's default when name is None.
    /// Loading and prediction execute on the caller's inference thread.
    pub fn function(&self, name: Option<&str>, units: ComputeUnits) -> Result<Function, Error> {
        autoreleasepool(|_| {
            let config = configuration(name, units);
            let url = NSURL::fileURLWithPath_isDirectory(
                &NSString::from_str(&self.path.to_string_lossy()),
                true,
            );
            // SAFETY: URL and configuration are live objects with valid values.
            let native =
                unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
                    .map_err(core_ml)?;
            // SAFETY: the model owns its feature description dictionaries.
            let description = unsafe {
                let d = native.modelDescription();
                FunctionDescription {
                    inputs: describe(&d.inputDescriptionsByName(), false)?,
                    outputs: describe(&d.outputDescriptionsByName(), false)?,
                    states: describe(&d.stateDescriptionsByName(), true)?,
                }
            };
            Ok(Function {
                native,
                description,
                not_sync: PhantomData,
            })
        })
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
}
/// One loaded Core ML function. It may be moved to an inference worker, but
/// calls on the same function cannot overlap.
pub struct Function {
    native: Retained<MLModel>,
    description: FunctionDescription,
    not_sync: PhantomData<Cell<()>>,
}
// SAFETY: MLModel permits use from any thread. Function is !Sync and all
// prediction calls require exclusive ownership, so calls never overlap.
unsafe impl Send for Function {}
impl Function {
    pub fn description(&self) -> &FunctionDescription {
        &self.description
    }
    pub fn predict(&mut self, inputs: &Inputs) -> Result<Prediction, Error> {
        autoreleasepool(|_| {
            if !self.description.states.is_empty() {
                return Err(Error::StateRequired);
            }
            let input = ProtocolObject::from_retained(inputs.provider()?);
            // SAFETY: valid feature provider; &mut self serializes model use.
            let features =
                unsafe { self.native.predictionFromFeatures_error(&input) }.map_err(core_ml)?;
            Ok(Prediction { features })
        })
    }
    pub fn predict_batch(&mut self, batch: &[Inputs]) -> Result<Vec<Prediction>, Error> {
        autoreleasepool(|_| {
            if !self.description.states.is_empty() {
                return Err(Error::StateRequired);
            }
            if batch.is_empty() {
                return Ok(Vec::new());
            }
            let providers = batch
                .iter()
                .map(|x| x.provider().map(ProtocolObject::from_retained))
                .collect::<Result<Vec<_>, _>>()?;
            let array = NSArray::from_retained_slice(&providers);
            // SAFETY: every array entry is a live feature provider.
            let provider = unsafe {
                MLArrayBatchProvider::initWithFeatureProviderArray(
                    MLArrayBatchProvider::alloc(),
                    &array,
                )
            };
            // SAFETY: exclusive model access and a valid batch provider.
            let out = unsafe {
                self.native
                    .predictionsFromBatch_error(&ProtocolObject::from_retained(provider))
            }
            .map_err(core_ml)?;
            // SAFETY: out is a retained Core ML batch provider.
            let count = unsafe { out.count() } as usize;
            if count != batch.len() {
                return Err(Error::BatchCount {
                    expected: batch.len(),
                    actual: count,
                });
            }
            Ok((0..count)
                .map(|i| Prediction {
                    // SAFETY: i is within the returned batch count.
                    features: unsafe { out.featuresAtIndex(i as NSInteger) },
                })
                .collect())
        })
    }
    /// Transfer this function to a session that exclusively owns its state.
    /// Load another function from the artifact for an independent request.
    pub fn into_session(self) -> Session {
        autoreleasepool(|_| {
            // SAFETY: newState creates zeroed state belonging to this exact model.
            let state = unsafe { self.native.newState() };
            Session {
                function: self,
                state,
            }
        })
    }
}
/// A function and its state, moved and reset together. State from another
/// model cannot be passed to a prediction through this API.
pub struct Session {
    function: Function,
    state: Retained<MLState>,
}
// SAFETY: Session owns both objects and requires &mut self for prediction
// and reset; no aliases to its mutable state are exposed.
unsafe impl Send for Session {}
impl Session {
    pub fn description(&self) -> &FunctionDescription {
        self.function.description()
    }
    pub fn reset(&mut self) {
        autoreleasepool(|_| {
            // SAFETY: replacement state is created by its owning model.
            self.state = unsafe { self.function.native.newState() };
        })
    }
    pub fn predict(&mut self, inputs: &Inputs) -> Result<Prediction, Error> {
        autoreleasepool(|_| {
            let input = ProtocolObject::from_retained(inputs.provider()?);
            // SAFETY: this state was created by this model, remains live, and
            // exclusive session access prevents overlapping state mutations.
            let features = unsafe {
                self.function
                    .native
                    .predictionFromFeatures_usingState_error(&input, &self.state)
            }
            .map_err(core_ml)?;
            Ok(Prediction { features })
        })
    }
}
/// Native outputs, retained until their last consumer releases them.
pub struct Prediction {
    features: Retained<ProtocolObject<dyn MLFeatureProvider>>,
}
impl Prediction {
    pub fn tensor<T: Element>(&self, name: &str) -> Result<Tensor<T>, Error> {
        autoreleasepool(|_| {
            // SAFETY: features is a live provider returned by Core ML.
            let value = unsafe { self.features.featureValueForName(&NSString::from_str(name)) }
                .ok_or_else(|| Error::MissingOutput(name.into()))?;
            // SAFETY: a non-array feature returns None.
            let array = unsafe { value.multiArrayValue() }
                .ok_or_else(|| Error::MissingOutput(name.into()))?;
            Tensor::from_array(array, name)
        })
    }
}
pub(crate) fn configuration(
    name: Option<&str>,
    units: ComputeUnits,
) -> Retained<MLModelConfiguration> {
    // SAFETY: native allocation and setters with valid argument values.
    unsafe {
        let config = MLModelConfiguration::new();
        config.setComputeUnits(units.ml());
        config.setFunctionName(name.map(NSString::from_str).as_deref());
        config
    }
}
fn describe(
    dictionary: &NSDictionary<NSString, MLFeatureDescription>,
    state: bool,
) -> Result<BTreeMap<String, FeatureDescription>, Error> {
    let mut features = BTreeMap::new();
    let (names, descriptions) = dictionary.to_vecs();
    for (name, feature) in names.into_iter().zip(descriptions) {
        // SAFETY: descriptions and constraints belong to the loaded model.
        let (dtype, shape, optional) = unsafe {
            if state {
                let c = feature
                    .stateConstraint()
                    .ok_or_else(|| Error::UnsupportedFeature(name.to_string()))?;
                (c.dataType(), c.bufferShape(), false)
            } else {
                let c = feature
                    .multiArrayConstraint()
                    .ok_or_else(|| Error::UnsupportedFeature(name.to_string()))?;
                (c.dataType(), c.shape(), feature.isOptional())
            }
        };
        use objc2_core_ml::MLMultiArrayDataType as D;
        let data_type = match dtype {
            D::Int32 => DataType::Int32,
            D::Float16 => DataType::Float16,
            D::Float32 => DataType::Float32,
            _ => return Err(Error::UnsupportedFeature(name.to_string())),
        };
        features.insert(
            name.to_string(),
            FeatureDescription {
                data_type,
                default_shape: shape.iter().map(|d| d.as_usize()).collect(),
                optional,
            },
        );
    }
    Ok(features)
}
pub(crate) fn compiled_bundle(dir: &Path) -> Result<PathBuf, Error> {
    let entries = std::fs::read_dir(dir).map_err(|source| Error::Io {
        path: dir.into(),
        source,
    })?;
    let mut bundles = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|source| Error::Io {
                path: dir.into(),
                source,
            })?
            .path();
        if path.extension().is_some_and(|e| e == "mlmodelc") {
            bundles.push(path);
        }
    }
    if bundles.len() != 1 {
        return Err(Error::ModelDir(dir.into()));
    }
    Ok(bundles.remove(0))
}
pub(crate) fn core_ml(error: Retained<NSError>) -> Error {
    Error::CoreMl {
        domain: error.domain().to_string(),
        code: error.code(),
        message: error.localizedDescription().to_string(),
    }
}
