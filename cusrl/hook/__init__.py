from .advantage import AdvantageNormalization, AdvantageReduction
from .condition import ConditionalObjectiveActivation
from .gae import GeneralizedAdvantageEstimation
from .gradient import GradientClipping
from .initialization import ModuleInitialization
from .lr_schedule import AdaptiveLRSchedule, MiniBatchWiseLRSchedule, ThresholdLRSchedule
from .normalization import ObservationNormalization
from .on_policy import OnPolicyPreparation
from .ppo import EntropyLoss, PpoSurrogateLoss
from .representation import (
    NextStatePrediction,
    ReturnPrediction,
    StatePrediction,
)
from .rnd import RandomNetworkDistillation
from .schedule import (
    HookActivationSchedule,
    OnPolicyBufferCapacitySchedule,
    ParameterSchedule,
)
from .smoothness import ActionSmoothnessLoss
from .statistics import OnPolicyStatistics
from .symmetry import (
    SymmetricArchitecture,
    SymmetricDataAugmentation,
    SymmetryLoss,
)
from .value import ValueComputation, ValueLoss

# alias
GAE = GeneralizedAdvantageEstimation
PPOSurrogateLoss = PpoSurrogateLoss

__all__ = [
    "ActionSmoothnessLoss",
    "AdaptiveLRSchedule",
    "AdvantageNormalization",
    "AdvantageReduction",
    "ConditionalObjectiveActivation",
    "EntropyLoss",
    "GAE",
    "GeneralizedAdvantageEstimation",
    "GradientClipping",
    "HookActivationSchedule",
    "MiniBatchWiseLRSchedule",
    "ModuleInitialization",
    "NextStatePrediction",
    "ObservationNormalization",
    "OnPolicyBufferCapacitySchedule",
    "OnPolicyPreparation",
    "OnPolicyStatistics",
    "ParameterSchedule",
    "PPOSurrogateLoss",
    "PpoSurrogateLoss",
    "RandomNetworkDistillation",
    "ReturnPrediction",
    "StatePrediction",
    "SymmetricArchitecture",
    "SymmetricDataAugmentation",
    "SymmetryLoss",
    "ThresholdLRSchedule",
    "ValueComputation",
    "ValueLoss",
]
