from .dmd import DMD
from .hodmd import HODMD

# temporary bug fix until completion of BagDMD
# from .bagdmd import BagDMD, HOBagDMD
from .optdmd import OptDMD
from .hooptdmd import HOOptDMD
from .mssa import MSSA, PMSSA
from .svd import (
    PODSubspaceDependencyResult,
    SVD,
    pod_subspace_data_dependency,
    subspace_similarity,
)

# from .linear_control import LinearControlModel
from .linear_model import LinearModel
from .dft import DFT, PDFT
from .spod import AMSPOD, PAMSPOD, mode_similarity
from .periodogram import AMPS

__all__ = [
    "AMPS",
    "AMSPOD",
    "curvilinear_surface_phase_velocity",
    "DFT",
    "DMD",
    "HODMD",
    "HOOptDMD",
    "LinearModel",
    "MSSA",
    "mode_similarity",
    "OptDMD",
    "PAMSPOD",
    "PDFT",
    "PMSSA",
    "pod_subspace_data_dependency",
    "PODSubspaceDependencyResult",
    "PSPExplorer",
    "SVD",
    "subspace_similarity",
]


def __getattr__(name: str):
    """Load data-dependent analysis tools only when requested."""
    if name == "PSPExplorer":
        from .psp_explorer import PSPExplorer

        globals()[name] = PSPExplorer
        return PSPExplorer
    if name == "curvilinear_surface_phase_velocity":
        from .propagation import curvilinear_surface_phase_velocity

        globals()[name] = curvilinear_surface_phase_velocity
        return curvilinear_surface_phase_velocity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
