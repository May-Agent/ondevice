//! The `manifest.json` a model directory ships beside its compiled model.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::Error;

/// The element type of a multi-array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    /// `int32`.
    Int32,
    /// `float16`.
    Float16,
    /// `float32`.
    Float32,
}

/// One dimension of a declared shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "RawDim")]
pub enum Dim {
    /// A fixed extent.
    Fixed(usize),
    /// The sequence length, `"length"` in the manifest: the bucket length of
    /// the function that runs.
    Length,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawDim {
    Fixed(usize),
    Symbol(String),
}

impl TryFrom<RawDim> for Dim {
    type Error = String;

    fn try_from(raw: RawDim) -> Result<Self, String> {
        match raw {
            RawDim::Fixed(extent) => Ok(Self::Fixed(extent)),
            RawDim::Symbol(symbol) if symbol == "length" => Ok(Self::Length),
            RawDim::Symbol(symbol) => Err(format!(
                "unknown dimension {symbol:?}: a dimension is a number or \"length\""
            )),
        }
    }
}

/// A declared input or output multi-array.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TensorSpec {
    /// The element type.
    #[serde(rename = "dtype")]
    pub data_type: DataType,
    /// The shape.
    pub shape: Vec<Dim>,
}

/// One function of the multifunction model: a bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    /// The Core ML function name, such as `len_128`.
    pub name: String,
    /// The sequence length the function accepts.
    pub len: usize,
}

/// A model directory's manifest: the model's functions, one per sequence
/// length bucket, and the multi-arrays every function shares.
///
/// Validated on load: there is at least one function, no two share a
/// length, every input is a sequence shaped `[1, …, 1, length]`, and there
/// is at least one output. Fields the manifest carries beyond these are the
/// model's own and are ignored here.
#[derive(Debug, Clone)]
pub struct Manifest {
    /// Ascending by `len`.
    functions: Box<[Function]>,
    inputs: BTreeMap<String, TensorSpec>,
    outputs: BTreeMap<String, TensorSpec>,
}

#[derive(Deserialize)]
struct RawManifest {
    functions: BTreeMap<String, usize>,
    inputs: BTreeMap<String, TensorSpec>,
    outputs: BTreeMap<String, TensorSpec>,
}

impl Manifest {
    /// Loads and validates `<dir>/manifest.json`.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] when it cannot be read, [`Error::Manifest`] when it is
    /// not a manifest, [`Error::ManifestContent`] when it fails validation.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = dir.as_ref().join("manifest.json");
        let bytes = std::fs::read(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        Self::parse(&bytes).map_err(|error| match error {
            Invalid::Json(source) => Error::Manifest { path, source },
            Invalid::Content(reason) => Error::ManifestContent { path, reason },
        })
    }

    fn parse(bytes: &[u8]) -> Result<Self, Invalid> {
        let raw: RawManifest = serde_json::from_slice(bytes).map_err(Invalid::Json)?;
        let mut functions: Vec<Function> = raw
            .functions
            .into_iter()
            .map(|(name, len)| Function { name, len })
            .collect();
        functions.sort_unstable_by_key(|function| function.len);
        if functions.is_empty() {
            return Err(Invalid::Content("no functions"));
        }
        if functions.first().is_some_and(|function| function.len == 0) {
            return Err(Invalid::Content("a function accepts length 0"));
        }
        if functions.windows(2).any(|pair| pair[0].len == pair[1].len) {
            return Err(Invalid::Content("two functions accept the same length"));
        }
        let sequence = |spec: &TensorSpec| {
            spec.shape.split_last().is_some_and(|(last, leading)| {
                *last == Dim::Length && leading.iter().all(|dim| *dim == Dim::Fixed(1))
            })
        };
        if !raw.inputs.values().all(sequence) {
            return Err(Invalid::Content("an input is not shaped [1, …, 1, length]"));
        }
        if raw.outputs.is_empty() {
            return Err(Invalid::Content("no outputs"));
        }
        Ok(Self {
            functions: functions.into_boxed_slice(),
            inputs: raw.inputs,
            outputs: raw.outputs,
        })
    }

    /// The functions, ascending by length.
    #[must_use]
    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    /// The largest sequence length any function accepts.
    #[must_use]
    pub fn max_len(&self) -> usize {
        self.functions[self.functions.len() - 1].len
    }

    /// The smallest function that holds `len`, or `None` past
    /// [`max_len`](Self::max_len).
    #[must_use]
    pub fn function_for(&self, len: usize) -> Option<&Function> {
        self.functions.iter().find(|function| len <= function.len)
    }

    /// The declared inputs by feature name.
    #[must_use]
    pub fn inputs(&self) -> &BTreeMap<String, TensorSpec> {
        &self.inputs
    }

    /// The declared outputs by feature name.
    #[must_use]
    pub fn outputs(&self) -> &BTreeMap<String, TensorSpec> {
        &self.outputs
    }
}

/// Why manifest bytes are not a valid manifest.
enum Invalid {
    Json(serde_json::Error),
    Content(&'static str),
}

#[cfg(test)]
mod tests {
    use super::{DataType, Dim, Invalid, Manifest};

    /// The manifest the EmbeddingGemma 2 Core ML conversion ships, with
    /// its model-specific fields.
    const EMBEDDING: &str = r#"{
        "model": "google/embeddinggemma-2",
        "functions": {"len_256": 256, "len_128": 128, "len_512": 512},
        "default_function": "len_128",
        "inputs": {
            "input_ids": {"dtype": "int32", "shape": [1, "length"]},
            "attention_mask": {"dtype": "float16", "shape": [1, "length"]}
        },
        "outputs": {"embedding": {"dtype": "float16", "shape": [1, 768]}},
        "embedding_dim": 768
    }"#;

    fn reason(json: &str) -> &'static str {
        match Manifest::parse(json.as_bytes()) {
            Err(Invalid::Content(reason)) => reason,
            Err(Invalid::Json(error)) => panic!("not JSON-valid: {error}"),
            Ok(_) => panic!("validation passed"),
        }
    }

    #[test]
    fn functions_are_ascending_and_buckets_pick_the_smallest_fit() {
        let manifest = Manifest::parse(EMBEDDING.as_bytes()).ok().unwrap();
        let lens: Vec<usize> = manifest.functions().iter().map(|f| f.len).collect();
        assert_eq!(lens, [128, 256, 512]);
        assert_eq!(manifest.max_len(), 512);
        assert_eq!(manifest.function_for(1).unwrap().name, "len_128");
        assert_eq!(manifest.function_for(128).unwrap().name, "len_128");
        assert_eq!(manifest.function_for(129).unwrap().name, "len_256");
        assert_eq!(manifest.function_for(512).unwrap().name, "len_512");
        assert!(manifest.function_for(513).is_none());
    }

    #[test]
    fn specs_parse_typed() {
        let manifest = Manifest::parse(EMBEDDING.as_bytes()).ok().unwrap();
        let ids = &manifest.inputs()["input_ids"];
        assert_eq!(ids.data_type, DataType::Int32);
        assert_eq!(ids.shape, [Dim::Fixed(1), Dim::Length]);
        let embedding = &manifest.outputs()["embedding"];
        assert_eq!(embedding.data_type, DataType::Float16);
        assert_eq!(embedding.shape, [Dim::Fixed(1), Dim::Fixed(768)]);
    }

    #[test]
    fn invalid_manifests_are_refused() {
        assert_eq!(
            reason(
                r#"{"functions": {}, "inputs": {}, "outputs": {"y": {"dtype": "float32", "shape": [1]}}}"#
            ),
            "no functions"
        );
        assert_eq!(
            reason(
                r#"{"functions": {"a": 8, "b": 8}, "inputs": {}, "outputs": {"y": {"dtype": "float32", "shape": [1]}}}"#
            ),
            "two functions accept the same length"
        );
        assert_eq!(
            reason(
                r#"{"functions": {"a": 8}, "inputs": {"x": {"dtype": "int32", "shape": ["length", 1]}}, "outputs": {"y": {"dtype": "float32", "shape": [1]}}}"#
            ),
            "an input is not shaped [1, …, 1, length]"
        );
        assert_eq!(
            reason(r#"{"functions": {"a": 8}, "inputs": {}, "outputs": {}}"#),
            "no outputs"
        );
    }

    #[test]
    fn unknown_types_and_dimensions_are_parse_errors() {
        for json in [
            r#"{"functions": {"a": 8}, "inputs": {"x": {"dtype": "int64", "shape": [1, "length"]}}, "outputs": {}}"#,
            r#"{"functions": {"a": 8}, "inputs": {"x": {"dtype": "int32", "shape": [1, "batch"]}}, "outputs": {}}"#,
        ] {
            assert!(matches!(
                Manifest::parse(json.as_bytes()),
                Err(Invalid::Json(_))
            ));
        }
    }
}
