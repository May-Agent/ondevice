use objc2_core_ml::MLComputeUnits;

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
    pub(crate) fn ml(self) -> MLComputeUnits {
        match self {
            Self::CpuOnly => MLComputeUnits::CPUOnly,
            Self::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            Self::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
            Self::All => MLComputeUnits::All,
        }
    }
}
