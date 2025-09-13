from .advantage import (
    AdvantageNormalization,
    AdvantageReduction,
)
from .amp import AdversarialMotionPrior
from .condition import ConditionalObjectiveActivation
from .distillation import PolicyDistillationLoss
from .environment_spec import (
    DynamicEnvironmentSpecOverride,
    EnvironmentSpecOverride,
)
from .gae import GeneralizedAdvantageEstimation
from .gradient import GradientClipping
from .initialization import ModuleInitialization
from .lr_schedule import (
    AdaptiveLRSchedule,
    MiniBatchWiseLRSchedule,
    ThresholdLRSchedule,
)
from .normalization import ObservationNormalization
from .on_policy import (
    OnPolicyBufferCapacitySchedule,
    OnPolicyPreparation,
    OnPolicyStatistics,
)
from .ppo import (
    EntropyLoss,
    PpoSurrogateLoss,
)
from .representation import (
    NextStatePrediction,
    ReturnPrediction,
    StatePrediction,
)
from .reward import RewardShaping
from .rnd import RandomNetworkDistillation
from .schedule import (
    HookActivationSchedule,
    HookParameterSchedule,
)
from .smoothness import ActionSmoothnessLoss
from .symmetry import (
    SymmetricArchitecture,
    SymmetricDataAugmentation,
    SymmetryLoss,
)
from .value import ValueComputation, ValueLoss

# alias for backward compatibility
GAE = GeneralizedAdvantageEstimation
PPOSurrogateLoss = PpoSurrogateLoss

__all__ = [
    "ActionSmoothnessLoss",
    "AdaptiveLRSchedule",
    "AdvantageNormalization",
    "AdvantageReduction",
    "AdversarialMotionPrior",
    "ConditionalObjectiveActivation",
    "DynamicEnvironmentSpecOverride",
    "EntropyLoss",
    "EnvironmentSpecOverride",
    "GAE",
    "GeneralizedAdvantageEstimation",
    "GradientClipping",
    "HookActivationSchedule",
    "HookParameterSchedule",
    "MiniBatchWiseLRSchedule",
    "ModuleInitialization",
    "NextStatePrediction",
    "ObservationNormalization",
    "OnPolicyBufferCapacitySchedule",
    "OnPolicyPreparation",
    "OnPolicyStatistics",
    "PolicyDistillationLoss",
    "PPOSurrogateLoss",
    "PpoSurrogateLoss",
    "RandomNetworkDistillation",
    "ReturnPrediction",
    "RewardShaping",
    "StatePrediction",
    "SymmetricArchitecture",
    "SymmetricDataAugmentation",
    "SymmetryLoss",
    "ThresholdLRSchedule",
    "ValueComputation",
    "ValueLoss",
]
