"""
ShEPhERD: Diffusing Shape, Electrostatics, and Pharmacophores for Drug Design
"""

__version__ = "0.1.0"

# shepherd model components
from shepherd.lightning_module import LightningModule
from shepherd.inference import inference_sample

# inference scaling utilities
from shepherd.inference_scaling import (
    ShepherdModelRunner,
    SAScoreVerifier,
    CLogPVerifier,
    MultiObjectiveVerifier,
    RandomSearch,
    ZeroOrderSearch,
    GuidedSearch,
)

# make components available at the package level
__all__ = [
    "LightningModule",
    "inference_sample",
    "ShepherdModelRunner",
    "SAScoreVerifier",
    "CLogPVerifier",
    "MultiObjectiveVerifier",
    "RandomSearch",
    "ZeroOrderSearch",
    "GuidedSearch",
]
