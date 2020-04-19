from .gcca import GCCA
from .omnibus import Omnibus
from .mvmds import MVMDS
try:
    from .splitae import SplitAE
    from .dcca import DCCA, linear_cca, cca_loss, MlpNet, DeepPairedNetworks
except ModuleNotFoundError as er:
    torch=False
from .kcca import KCCA
from .utils import select_dimension

__all__ = [
        "GCCA",
        "Omnibus",
        "MVMDS",
        "KCCA",
        "select_dimension",
    ]
if 'torch' not in locals():
    __all__ += [
        "SplitAE",
        "DCCA",
    ]
