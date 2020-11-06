from .gcca import GCCA
from .omnibus import Omnibus
from .mvmds import MVMDS
from .kcca import KCCA
from .mcca import MCCA
from .kmcca import KMCCA
from .utils import select_dimension
from .cca import CCA

try:
    import torch  # noqa
    from torch.utils.data import Dataset, DataLoader  # noqa
    import torch.nn as nn  # noqa
    from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler  # noqa
    from .dcca import DCCA, linear_cca, cca_loss, MlpNet, DeepPairedNetworks  # noqa
    from .splitae import SplitAE  # noqa
except ModuleNotFoundError:
    pass

__all__ = [
        "CCA",
        "GCCA",
        "Omnibus",
        "MVMDS",
        "KCCA",
        "MCCA",
        "KMCCA",
        "select_dimension",
        "DCCA",
        "SplitAE",
        "linear_cca",
        "cca_loss",
        "MlpNet",
        "DeepPairedNetworks",
    ]
