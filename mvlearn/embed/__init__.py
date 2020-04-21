import sys
from .gcca import GCCA
from .omnibus import Omnibus
from .mvmds import MVMDS
from .kcca import KCCA
from .utils import select_dimension

__all__ = [
        "GCCA",
        "Omnibus",
        "MVMDS",
        "KCCA",
        "select_dimension",
    ]
if 'torch' in sys.modules:
    from .dcca import DCCA, linear_cca, cca_loss, MlpNet, DeepPairedNetworks
    __all__ += ["DCCA"]
if 'torch' in sys.modules and 'torchvision' in sys.modules:
    from .splitae import SplitAE
    __all__ += ["SplitAE"]

    
