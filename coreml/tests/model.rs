//! Model-dependent checks, gated on `ONDEVICE_COREML_MODEL_DIR` naming a
//! model directory (`manifest.json` beside one `*.mlmodelc`). The model is
//! the fixture: any sequence model whose `int32` inputs take small token ids
//! and whose float inputs take an attention mask of ones, such as the
//! EmbeddingGemma 2 conversion repo-search publishes. They fail loudly when
//! the variable is unset.

use std::path::PathBuf;

use ondevice_coreml::{
    ComputeUnits, DataType, Dim, Error, Input, Manifest, Model, Prediction, Sequence, f16,
};

fn model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("ONDEVICE_COREML_MODEL_DIR")
            .expect("ONDEVICE_COREML_MODEL_DIR must name a model directory"),
    )
}

/// The values of every declared input for a sequence of `len`, seeded by
/// `seed` so different sequences differ.
struct Values {
    int32: Vec<(String, Vec<i32>)>,
    float16: Vec<(String, Vec<f16>)>,
    float32: Vec<(String, Vec<f32>)>,
}

impl Values {
    fn new(manifest: &Manifest, len: usize, seed: usize) -> Self {
        let mut values = Self {
            int32: Vec::new(),
            float16: Vec::new(),
            float32: Vec::new(),
        };
        for (name, spec) in manifest.inputs() {
            let name = name.clone();
            match spec.data_type {
                DataType::Int32 => values.int32.push((
                    name,
                    (0..len)
                        .map(|i| 2 + ((i * 7 + seed * 13) % 500) as i32)
                        .collect(),
                )),
                DataType::Float16 => values.float16.push((name, vec![f16::ONE; len])),
                DataType::Float32 => values.float32.push((name, vec![1.0; len])),
            }
        }
        values
    }

    fn sequences(&self) -> Vec<Box<dyn Input + '_>> {
        let mut inputs: Vec<Box<dyn Input + '_>> = Vec::new();
        for (name, values) in &self.int32 {
            inputs.push(Box::new(Sequence {
                name,
                values,
                pad: 0,
            }));
        }
        for (name, values) in &self.float16 {
            inputs.push(Box::new(Sequence {
                name,
                values,
                pad: f16::ZERO,
            }));
        }
        for (name, values) in &self.float32 {
            inputs.push(Box::new(Sequence {
                name,
                values,
                pad: 0.0,
            }));
        }
        inputs
    }
}

fn predict(model: &mut Model, len: usize, seed: usize) -> Prediction {
    let values = Values::new(model.manifest(), len, seed);
    let inputs = values.sequences();
    let inputs: Vec<&dyn Input> = inputs.iter().map(|input| &**input).collect();
    model.predict(&inputs).unwrap()
}

/// Every output of `prediction`, flattened to `f32` in name order.
fn outputs(manifest: &Manifest, prediction: &Prediction) -> Vec<f32> {
    manifest
        .outputs()
        .iter()
        .flat_map(|(name, spec)| match spec.data_type {
            DataType::Int32 => prediction
                .output::<i32>(name)
                .unwrap()
                .into_data()
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>(),
            DataType::Float16 => prediction
                .output::<f16>(name)
                .unwrap()
                .into_data()
                .into_iter()
                .map(f32::from)
                .collect(),
            DataType::Float32 => prediction.output::<f32>(name).unwrap().into_data(),
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "output lengths differ");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
#[ignore = "needs ONDEVICE_COREML_MODEL_DIR; run with --ignored"]
fn every_bucket_runs_and_returns_the_declared_outputs() {
    let mut model = Model::open(model_dir(), ComputeUnits::CpuAndNeuralEngine).unwrap();
    let manifest = model.manifest().clone();
    let mut shortest = 1;
    for function in manifest.functions() {
        // The shortest and the longest input each bucket takes.
        for len in [shortest, function.len] {
            let prediction = predict(&mut model, len, len);
            for (name, spec) in manifest.outputs() {
                let fixed: Option<Vec<usize>> = spec
                    .shape
                    .iter()
                    .map(|dim| match dim {
                        Dim::Fixed(extent) => Some(*extent),
                        Dim::Length => None,
                    })
                    .collect();
                let count = match spec.data_type {
                    DataType::Int32 => prediction.output::<i32>(name).unwrap().data().len(),
                    DataType::Float16 => prediction.output::<f16>(name).unwrap().data().len(),
                    DataType::Float32 => prediction.output::<f32>(name).unwrap().data().len(),
                };
                if let Some(fixed) = fixed {
                    assert_eq!(count, fixed.iter().product::<usize>(), "{name} at {len}");
                }
            }
        }
        shortest = function.len + 1;
    }
}

#[test]
#[ignore = "needs ONDEVICE_COREML_MODEL_DIR; run with --ignored"]
fn an_input_over_the_largest_bucket_is_too_long() {
    let mut model = Model::open(model_dir(), ComputeUnits::CpuAndNeuralEngine).unwrap();
    let max = model.manifest().max_len();
    let values = Values::new(model.manifest(), max + 1, 0);
    let inputs = values.sequences();
    let inputs: Vec<&dyn Input> = inputs.iter().map(|input| &**input).collect();
    match model.predict(&inputs) {
        Err(Error::TooLong { len, max: cap }) => {
            assert_eq!((len, cap), (max + 1, max));
        }
        Err(error) => panic!("expected TooLong, got {error}"),
        Ok(_) => panic!("{} is over the largest bucket", max + 1),
    }
}

#[test]
#[ignore = "needs ONDEVICE_COREML_MODEL_DIR; run with --ignored"]
fn undeclared_inputs_and_mistyped_outputs_are_errors() {
    let mut model = Model::open(model_dir(), ComputeUnits::CpuAndNeuralEngine).unwrap();
    let ids = [2, 3, 4];
    let undeclared = Sequence {
        name: "no such input",
        values: &ids,
        pad: 0,
    };
    assert!(matches!(
        model.predict(&[&undeclared]),
        Err(Error::UnknownInput(name)) if name == "no such input"
    ));
    let prediction = predict(&mut model, 3, 0);
    let (name, spec) = model.manifest().outputs().iter().next().unwrap();
    let mistyped = match spec.data_type {
        DataType::Int32 => prediction.output::<f32>(name).map(drop),
        DataType::Float16 | DataType::Float32 => prediction.output::<i32>(name).map(drop),
    };
    assert!(
        matches!(mistyped, Err(Error::OutputType { .. })),
        "{mistyped:?}"
    );
    assert!(matches!(
        prediction.output::<f32>("no such output"),
        Err(Error::MissingOutput(_))
    ));
}

#[test]
#[ignore = "needs ONDEVICE_COREML_MODEL_DIR; run with --ignored"]
fn a_batch_equals_its_single_predictions() {
    let mut model = Model::open(model_dir(), ComputeUnits::CpuAndNeuralEngine).unwrap();
    let manifest = model.manifest().clone();
    // Eight sequences of different lengths, all in the smallest bucket.
    let lens: Vec<usize> = (0..8).map(|i| 4 + i * 3).collect();
    let values: Vec<Values> = lens
        .iter()
        .enumerate()
        .map(|(seed, &len)| Values::new(&manifest, len, seed))
        .collect();
    let boxed: Vec<Vec<Box<dyn Input + '_>>> = values.iter().map(Values::sequences).collect();
    let items: Vec<Vec<&dyn Input>> = boxed
        .iter()
        .map(|inputs| inputs.iter().map(|input| &**input).collect())
        .collect();
    let batch: Vec<&[&dyn Input]> = items.iter().map(Vec::as_slice).collect();

    let batched = model.predict_batch(&batch).unwrap();

    assert_eq!(batched.len(), batch.len());
    for (i, (prediction, item)) in batched.iter().zip(&batch).enumerate() {
        let single = model.predict(item).unwrap();
        let c = cosine(
            &outputs(&manifest, prediction),
            &outputs(&manifest, &single),
        );
        assert!(c >= 0.9999, "item {i}: cosine {c}");
    }
    assert!(model.predict_batch(&[]).unwrap().is_empty());
}

#[cfg(feature = "headroom")]
#[test]
#[ignore = "needs ONDEVICE_COREML_MODEL_DIR; run with --ignored"]
fn neural_engine_and_gpu_models_run_concurrently_and_agree() {
    const ITERATIONS: usize = 30;
    if let Err(blocked) = ondevice_coreml::gpu_headroom() {
        eprintln!("no GPU headroom ({blocked}); the GPU half is skipped");
        return;
    }
    let dir = model_dir();
    // Both threads finish `open` first, then release together so their
    // prediction loops overlap.
    let barrier = std::sync::Barrier::new(2);
    let run = |units: ComputeUnits| {
        let mut model = Model::open(&dir, units).unwrap();
        let manifest = model.manifest().clone();
        barrier.wait();
        (0..ITERATIONS)
            .map(|_| outputs(&manifest, &predict(&mut model, 16, 1)))
            .collect::<Vec<_>>()
    };
    let (ane, gpu) = std::thread::scope(|s| {
        let ane = s.spawn(|| run(ComputeUnits::CpuAndNeuralEngine));
        let gpu = s.spawn(|| run(ComputeUnits::CpuAndGpu));
        (ane.join().unwrap(), gpu.join().unwrap())
    });
    let c = cosine(&ane[0], &gpu[0]);
    assert!(c > 0.99, "Neural Engine vs GPU cosine {c}");
}
