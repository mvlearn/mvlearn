from .split import SimpleSplitter
from .merge import ConcatMerger, AverageMerger
from .random_gaussian_projection import RandomGaussianProjection
from .rsm import RandomSubspaceMethod
from .wrap import ViewClassifier

__all__ = [
    "SimpleSplitter",
    "ConcatMerger",
    "AverageMerger",
    "RandomGaussianProjection",
    "RandomSubspaceMethod",
    "ViewClassifier"
    ]
