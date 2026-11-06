//! Bucketed Core ML inference over one compiled multifunction model.
//!
//! A sequence model compiled for a few fixed sequence lengths ships as one
//! `.mlmodelc` with one function per length (a bucket), all sharing the
//! weights, beside a `manifest.json` naming the functions and the
//! multi-arrays they take and return. A [`Model`] loads every function onto
//! the chosen [`ComputeUnits`], pads typed [`Sequence`] inputs to the
//! smallest bucket that holds them, and returns typed [`Tensor`] outputs.
//! What the inputs mean — tokens, masks — and what the outputs mean is the
//! caller's.
//!
//! The `headroom` feature adds [`gpu_headroom`] and [`HeadroomMonitor`]:
//! whether the host is on mains power and thermally able to run GPU
//! inference, for a caller that only uses the GPU when it is.
//!
//! The crate is empty off Apple platforms.
#![cfg(target_vendor = "apple")]

use std::io;
use std::path::PathBuf;

mod manifest;
mod model;
#[cfg(feature = "headroom")]
mod power;
mod tensor;

pub use half::f16;
pub use manifest::{DataType, Dim, Function, Manifest, TensorSpec};
pub use model::{ComputeUnits, Model, Prediction};
pub use objc2_core_ml::MLMultiArrayDataType;
#[cfg(feature = "headroom")]
pub use power::{GpuBlocked, HeadroomMonitor, gpu_headroom};
pub use tensor::{Element, Input, Sequence, Tensor};

/// A [`Manifest`] or [`Model`] failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
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
