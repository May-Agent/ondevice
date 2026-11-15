//! Smallest-fit sequence padding over general model functions.
use crate::model::compiled_bundle;
use crate::{
    ComputeUnits, DataType, Error, Function, Input, Inputs, Manifest, Model, Prediction, Tensor,
    f16,
};
use std::path::Path;

pub struct BucketModel {
    manifest: Manifest,
    functions: Vec<Function>,
}
impl BucketModel {
    pub fn open(dir: impl AsRef<Path>, units: ComputeUnits) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let manifest = Manifest::load(dir)?;
        let model = Model::open(compiled_bundle(dir)?)?;
        let functions = manifest
            .functions()
            .iter()
            .map(|f| model.function(Some(&f.name), units))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            manifest,
            functions,
        })
    }
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn predict(&mut self, inputs: &[&dyn Input]) -> Result<Prediction, Error> {
        let index = self.bucket(sequence_len(inputs)?)?;
        let inputs = prepare(&self.manifest, inputs, self.manifest.functions()[index].len)?;
        self.functions[index].predict(&inputs)
    }
    pub fn predict_batch(&mut self, batch: &[&[&dyn Input]]) -> Result<Vec<Prediction>, Error> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let longest = batch
            .iter()
            .map(|x| sequence_len(x))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap();
        let index = self.bucket(longest)?;
        let len = self.manifest.functions()[index].len;
        let inputs = batch
            .iter()
            .map(|x| prepare(&self.manifest, x, len))
            .collect::<Result<Vec<_>, _>>()?;
        self.functions[index].predict_batch(&inputs)
    }
    fn bucket(&self, len: usize) -> Result<usize, Error> {
        self.manifest
            .functions()
            .iter()
            .position(|f| f.len >= len)
            .ok_or(Error::TooLong {
                len,
                max: self.manifest.max_len(),
            })
    }
}
fn sequence_len(inputs: &[&dyn Input]) -> Result<usize, Error> {
    let len = inputs.first().ok_or(Error::SequenceLengths)?.len();
    if inputs.iter().any(|i| i.len() != len) {
        return Err(Error::SequenceLengths);
    }
    Ok(len)
}
fn prepare(manifest: &Manifest, inputs: &[&dyn Input], len: usize) -> Result<Inputs, Error> {
    let mut result = Inputs::new();
    for input in inputs {
        let spec = manifest
            .inputs()
            .get(input.name())
            .ok_or_else(|| Error::UnknownInput(input.name().into()))?;
        if spec.data_type != input.data_type() {
            return Err(Error::InputType(input.name().into()));
        }
        let mut shape = vec![1; spec.shape.len()];
        *shape.last_mut().unwrap() = len;
        macro_rules! insert {
            ($t:ty, $zero:expr) => {{
                let mut values = vec![$zero; len];
                // SAFETY: values has len elements of the input's checked type;
                // the bucket is at least the sequence length and stride is one.
                unsafe {
                    input.fill(
                        std::ptr::NonNull::new(values.as_mut_ptr().cast()).unwrap(),
                        1,
                        len,
                    )
                };
                result.insert(input.name(), &Tensor::<$t>::from_slice(&shape, &values)?)?;
            }};
        }
        match spec.data_type {
            DataType::Int32 => insert!(i32, 0),
            DataType::Float16 => insert!(f16, f16::ZERO),
            DataType::Float32 => insert!(f32, 0.0),
        }
    }
    Ok(result)
}
