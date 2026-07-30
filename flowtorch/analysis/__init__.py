from .dmd import DMD
from .hodmd import HODMD

# temporary bug fix until completion of BagDMD
# from .bagdmd import BagDMD, HOBagDMD
from .optdmd import OptDMD
from .hooptdmd import HOOptDMD
from .mssa import MSSA, PMSSA
from .svd import SVD

# from .linear_control import LinearControlModel
from .linear_model import LinearModel
from .dft import DFT, PDFT
from .spod import AMSPOD, PAMSPOD
from .periodogram import AMPS

__all__ = [
    "AMPS",
    "AMSPOD",
    "DFT",
    "DMD",
    "HODMD",
    "HOOptDMD",
    "LinearModel",
    "MSSA",
    "OptDMD",
    "PAMSPOD",
    "PDFT",
    "PMSSA",
    "PSPExplorer",
    "SVD",
]


def __getattr__(name: str):
    """Load data-dependent analysis tools only when requested."""
    if name == "PSPExplorer":
        from .psp_explorer import PSPExplorer

        globals()[name] = PSPExplorer
        return PSPExplorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
