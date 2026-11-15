//! Core ML's compilation plan, not a measurement of runtime execution.
use crate::model::{configuration, core_ml};
use crate::{ComputeUnits, Error, Model};
use block2::RcBlock;
use objc2::{
    rc::Retained,
    runtime::{AnyObject, ProtocolObject},
};
use objc2_core_ml::{
    MLComputeDeviceProtocol, MLComputePlan, MLComputePlanDeviceUsage, MLModelStructure,
    MLModelStructureProgramBlock,
};
use objc2_foundation::{NSError, NSString, NSURL};

/// Compiler-reported placement and optional relative cost of one operation.
#[derive(Debug, Clone)]
pub struct OperationPlan {
    /// Function and nested block/layer indices identifying this operation.
    pub location: Vec<String>,
    pub operator: String,
    /// Native compute device class names; unknown placement remains None.
    pub preferred_device: Option<String>,
    pub supported_devices: Vec<String>,
    /// Core ML's relative estimated cost, not elapsed time.
    pub estimated_cost: Option<f64>,
}
#[derive(Debug, Clone)]
pub struct ComputePlan {
    pub requested_units: ComputeUnits,
    pub operations: Vec<OperationPlan>,
}
struct PlanHandle(Retained<MLComputePlan>);
// SAFETY: MLComputePlan is immutable metadata. The callback transfers its
// retained handle to one receiver; no mutable native state is shared.
unsafe impl Send for PlanHandle {}
impl Model {
    /// Asynchronously query Core ML's planned operation placement. Cancellation
    /// drops the receiver; the native callback still releases its retained data.
    pub async fn compute_plan(
        &self,
        function: Option<&str>,
        units: ComputeUnits,
    ) -> Result<ComputePlan, Error> {
        let url = NSURL::fileURLWithPath_isDirectory(
            &NSString::from_str(&self.path().to_string_lossy()),
            true,
        );
        let config = configuration(function, units);
        let (send, receive) = async_channel::bounded(1);
        let callback = RcBlock::new(move |plan: *mut MLComputePlan, error: *mut NSError| {
            // SAFETY: callback pointers are either nil or live borrowed objects;
            // retaining before returning transfers their lifetime to the receiver.
            let result = unsafe {
                match (Retained::retain(plan), Retained::retain(error)) {
                    (Some(plan), _) => Ok(PlanHandle(plan)),
                    (_, Some(error)) => Err(core_ml(error)),
                    _ => Err(Error::MissingCallbackResult),
                }
            };
            let _ = send.try_send(result);
        });
        // SAFETY: Core ML retains the completion block and its arguments for the
        // operation; the callback does not borrow this stack frame.
        unsafe {
            MLComputePlan::loadContentsOfURL_configuration_completionHandler(
                &url, &config, &callback,
            )
        };
        let plan = receive
            .recv()
            .await
            .map_err(|_| Error::MissingCallbackResult)??;
        let mut operations = Vec::new();
        // SAFETY: the retained plan owns its model structure.
        structure(
            &plan.0,
            &*unsafe { plan.0.modelStructure() },
            &[],
            &mut operations,
        )?;
        Ok(ComputePlan {
            requested_units: units,
            operations,
        })
    }
}
fn device_name(device: &ProtocolObject<dyn MLComputeDeviceProtocol>) -> String {
    let object: &AnyObject = device.as_ref();
    object.class().name().to_string_lossy().into_owned()
}
fn placement(usage: Option<Retained<MLComputePlanDeviceUsage>>) -> (Option<String>, Vec<String>) {
    match usage {
        Some(usage) => {
            // SAFETY: usage is immutable metadata retained from a compute plan.
            unsafe {
                (
                    Some(device_name(&usage.preferredComputeDevice())),
                    usage
                        .supportedComputeDevices()
                        .iter()
                        .map(|d| device_name(&d))
                        .collect(),
                )
            }
        }
        None => (None, Vec::new()),
    }
}
fn structure(
    plan: &MLComputePlan,
    model: &MLModelStructure,
    path: &[String],
    rows: &mut Vec<OperationPlan>,
) -> Result<(), Error> {
    // SAFETY: all queried structure objects belong to the retained plan.
    unsafe {
        if let Some(program) = model.program() {
            let (names, functions) = program.functions().to_vecs();
            for (name, function) in names.into_iter().zip(functions) {
                let mut path = path.to_vec();
                path.push(name.to_string());
                block(plan, &function.block(), &path, rows);
            }
        } else if let Some(network) = model.neuralNetwork() {
            for layer in network.layers().iter() {
                let mut location = path.to_vec();
                location.push(layer.name().to_string());
                let (preferred_device, supported_devices) =
                    placement(plan.computeDeviceUsageForNeuralNetworkLayer(&layer));
                rows.push(OperationPlan {
                    location,
                    operator: layer.r#type().to_string(),
                    preferred_device,
                    supported_devices,
                    estimated_cost: None,
                });
            }
        } else if let Some(pipeline) = model.pipeline() {
            for (name, submodel) in pipeline
                .subModelNames()
                .iter()
                .zip(pipeline.subModels().iter())
            {
                let mut path = path.to_vec();
                path.push(name.to_string());
                structure(plan, &submodel, &path, rows)?;
            }
        } else {
            return Err(Error::UnsupportedPlan);
        }
    }
    Ok(())
}
fn block(
    plan: &MLComputePlan,
    b: &MLModelStructureProgramBlock,
    path: &[String],
    rows: &mut Vec<OperationPlan>,
) {
    // SAFETY: the block, its operations and costs belong to the retained plan.
    unsafe {
        for (index, operation) in b.operations().iter().enumerate() {
            let mut location = path.to_vec();
            location.push(index.to_string());
            let (preferred_device, supported_devices) =
                placement(plan.computeDeviceUsageForMLProgramOperation(&operation));
            let estimated_cost = plan
                .estimatedCostOfMLProgramOperation(&operation)
                .map(|c| c.weight());
            rows.push(OperationPlan {
                location: location.clone(),
                operator: operation.operatorName().to_string(),
                preferred_device,
                supported_devices,
                estimated_cost,
            });
            for (index, nested) in operation.blocks().iter().enumerate() {
                let mut nested_path = location.clone();
                nested_path.push(index.to_string());
                block(plan, &nested, &nested_path, rows);
            }
        }
    }
}
