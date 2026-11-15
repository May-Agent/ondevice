//! Typed Core ML model functions, native tensors and stateful sessions.
//! Sequence padding and batching are a separate policy in [`BucketModel`].
#![cfg(target_vendor = "apple")]

use std::io;
use std::path::PathBuf;

mod bucket;
mod compute;
mod manifest;
mod model;
mod native;
#[cfg(feature = "plan")]
mod plan;
#[cfg(feature = "plan")]
pub use plan::{ComputePlan, OperationPlan};
#[cfg(feature = "headroom")]
mod power;
mod tensor;

pub use bucket::BucketModel;
pub use compute::ComputeUnits;
pub use half::f16;
pub use manifest::{Bucket, DataType, Dim, Manifest, TensorSpec};
pub use model::{FeatureDescription, Function, FunctionDescription, Model, Prediction, Session};
pub use native::{Inputs, Tensor};
pub use objc2_core_ml::MLMultiArrayDataType;
#[cfg(feature = "headroom")]
pub use power::{GpuBlocked, HeadroomMonitor, gpu_headroom};
pub use tensor::{Element, Input, Sequence, TensorData};

/// A [`Manifest`] or [`Model`] failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Core ML callback returned neither result nor error")]
    MissingCallbackResult,
    #[error("Core ML compute plan has no supported model structure")]
    UnsupportedPlan,
    #[error("tensor shape {shape:?} does not hold {elements} elements")]
    TensorShape { shape: Vec<usize>, elements: usize },
    #[error("duplicate input {0}")]
    DuplicateInput(String),
    #[error("input type disagrees with the manifest: {0}")]
    InputType(String),
    #[error("sequence inputs must be nonempty and have equal lengths")]
    SequenceLengths,
    #[error("stateful function requires a session")]
    StateRequired,
    #[error("unsupported feature {0}: expected a typed multi-array")]
    UnsupportedFeature(String),
    #[error("Core ML returned {actual} batch items, expected {expected}")]
    BatchCount { expected: usize, actual: usize },
    /// A file in the model directory could not be read.
    #[error("cannot read {}: {source}", path.display())]
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The I/O failure.
        source: io::Error,
    },
    /// `manifest.json` is not a manifest.
    #[error("{}: {source}", path.display())]
    Manifest {
        /// The manifest path.
        path: PathBuf,
        /// The parse failure.
        source: serde_json::Error,
    },
    /// `manifest.json` parsed but describes no valid model.
    #[error("{}: {reason}", path.display())]
    ManifestContent {
        /// The manifest path.
        path: PathBuf,
        /// What is wrong with it.
        reason: &'static str,
    },
    /// The model directory holds no `*.mlmodelc`, or several.
    #[error("{}: expected exactly one .mlmodelc bundle", .0.display())]
    ModelDir(PathBuf),
    /// Core ML returned an `NSError`.
    #[error("{domain} error {code}: {message}")]
    CoreMl {
        /// The error's domain.
        domain: String,
        /// The error's code.
        code: isize,
        /// `localizedDescription`.
        message: String,
    },
    /// The input is longer than the largest bucket.
    #[error("input of length {len} is over the largest bucket, {max}")]
    TooLong {
        /// The input's length.
        len: usize,
        /// The largest bucket's length.
        max: usize,
    },
    /// An input the manifest does not declare.
    #[error("the manifest declares no input {0:?}")]
    UnknownInput(String),
    /// The prediction has no multi-array output of this name.
    #[error("the prediction has no multi-array output {0:?}")]
    MissingOutput(String),
    /// The output's elements are not the requested type.
    #[error("output {name:?} holds {actual:?}, not {expected:?}")]
    OutputType {
        /// The output's name.
        name: String,
        /// The requested element type.
        expected: DataType,
        /// The multi-array's element type.
        actual: MLMultiArrayDataType,
    },
}
