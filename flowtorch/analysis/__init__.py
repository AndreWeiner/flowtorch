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
from .state_vector import (
    DataloaderStateVectorSource,
    FieldSpec,
    StateVectorLayout,
    StateVectorResult,
    StateVectorSource,
)
from .statistics import (
    DEFAULT_QUANTILES,
    STATISTIC_NAMES,
    DistributedExecution,
    HistogramResult,
    LinearTrendResult,
    MOMENT_NAMES,
    MomentDependencyResult,
    MomentFields,
    RunningMoments,
    SnapshotStatisticsResult,
    SpatialStatisticsResult,
    detect_linear_trend,
    linear_trend,
    moment_data_dependency,
    snapshot_statistics,
    spatial_statistics,
    spatiotemporal_histogram,
    statistical_moments,
)

__all__ = [
    "AMPS",
    "AMSPOD",
    "curvilinear_surface_phase_velocity",
    "DEFAULT_QUANTILES",
    "DistributedExecution",
    "DFT",
    "DMD",
    "DataloaderStateVectorSource",
    "FieldSpec",
    "HODMD",
    "HOOptDMD",
    "HistogramResult",
    "LinearModel",
    "LinearTrendResult",
    "MOMENT_NAMES",
    "MSSA",
    "mode_similarity",
    "moment_data_dependency",
    "MomentDependencyResult",
    "MomentFields",
    "OptDMD",
    "PAMSPOD",
    "PDFT",
    "PMSSA",
    "pod_subspace_data_dependency",
    "PODSubspaceDependencyResult",
    "PSPExplorer",
    "RunningMoments",
    "SVD",
    "snapshot_statistics",
    "SnapshotStatisticsResult",
    "spatial_statistics",
    "SpatialStatisticsResult",
    "spatiotemporal_histogram",
    "statistical_moments",
    "STATISTIC_NAMES",
    "StateVectorLayout",
    "StateVectorResult",
    "StateVectorSource",
    "subspace_similarity",
    "detect_linear_trend",
    "linear_trend",
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
