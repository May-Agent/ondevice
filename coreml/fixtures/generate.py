"""Generate small executable Core ML contracts for the Rust integration tests."""
from pathlib import Path
import json
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

ROOT = Path(__file__).parent / "models"
ROOT.mkdir(exist_ok=True)


def save(program, name):
    model = ct.convert(program, minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT32,
                       compute_units=ct.ComputeUnit.CPU_ONLY)
    path = ROOT / f"{name}.mlpackage"
    model.save(str(path))
    return path


def compile_model(path, destination):
    ct.models.utils.compile_model(str(path), destination_path=str(destination))


@mb.program(input_specs=[mb.TensorSpec((2, 3)), mb.TensorSpec((3,))],
            opset_version=ct.target.macOS15)
def affine(matrix, bias):
    return mb.add(x=matrix, y=bias, name="output")


@mb.program(input_specs=[mb.TensorSpec((2, 3), dtype=types.fp16),
                        mb.StateTensorSpec((2, 3), dtype=types.fp16)],
            opset_version=ct.target.macOS15)
def accumulate(delta, memory):
    value = mb.add(x=mb.read_state(input=memory), y=delta)
    return mb.coreml_update_state(state=memory, value=value, name="output")


compile_model(save(affine, "affine"), ROOT / "affine.mlmodelc")
compile_model(save(accumulate, "accumulate"), ROOT / "accumulate.mlmodelc")
descriptor = ct.utils.MultiFunctionDescriptor()
for length in (32, 64):
    @mb.program(input_specs=[mb.TensorSpec((1, length), dtype=types.int32)],
                opset_version=ct.target.macOS15)
    def bucket(tokens):
        return mb.reduce_sum(x=mb.cast(x=tokens, dtype="fp32"), axes=[1],
                             keep_dims=True, name="output")
    name = f"len_{length}"
    descriptor.add_function(str(save(bucket, name)), "main", name)
descriptor.default_function_name = "len_32"
package = ROOT / "buckets.mlpackage"
ct.utils.save_multifunction(descriptor, str(package))
directory = ROOT / "buckets"
directory.mkdir(exist_ok=True)
compile_model(package, directory / "model.mlmodelc")
(directory / "manifest.json").write_text(json.dumps({
    "functions": {"len_32": 32, "len_64": 64},
    "inputs": {"tokens": {"dtype": "int32", "shape": [1, "length"]}},
    "outputs": {"output": {"dtype": "float32", "shape": [1, 1]}}
}))
