from .gcca import GCCA
from .omnibus import Omnibus
from .pls import partial_least_squares_embedding
from .mvmds import MVMDS
from .splitae import SplitAE
from .kcca import KCCA

__all__ = ["GCCA", "Omnibus", "partial_least_squares_embedding",
           "MVMDS", "SplitAE", "KCCA"]
