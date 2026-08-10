import pytest
import torch as pt

from flowtorch.data import ImageMaskDiagnostics, create_image_mask


def test_extreme_filter_uses_independent_optional_bounds():
    data = pt.zeros((7, 7, 3))
    data[1, 1, 0] = -2.0
    data[2, 2, 1] = 3.0
    data[3, 3, 2] = float("nan")

    keep, diagnostics = create_image_mask(data, lower_bound=-1.0, threshold=1.0e12)

    assert not keep[1, 1]
    assert keep[2, 2]
    assert not keep[3, 3]
    assert diagnostics.extreme_count[1, 1] == 1
    assert diagnostics.extreme_count[3, 3] == 1

    keep, _ = create_image_mask(data, upper_bound=2.0, threshold=1.0e12)
    assert keep[1, 1]
    assert not keep[2, 2]
    assert not keep[3, 3]


def test_spatial_filter_detects_isolated_defect_in_single_image():
    data = pt.zeros((9, 9))
    data[4, 4] = 1.0

    keep, diagnostics = create_image_mask(data, threshold=5.0)

    assert not keep[4, 4]
    assert diagnostics.spatial_mask[4, 4]
    assert not diagnostics.extreme_mask.any()
    assert isinstance(diagnostics, ImageMaskDiagnostics)


def test_temporal_reductions_differ_and_chunking_is_invariant():
    data = pt.zeros((9, 9, 10))
    data[4, 4, 0] = 1.0

    _, maximum = create_image_mask(
        data, threshold=1.0e12, temporal_reduction="max", chunk_size=3
    )
    _, mean = create_image_mask(
        data, threshold=1.0e12, temporal_reduction="mean", chunk_size=4
    )
    _, unchunked = create_image_mask(
        data, threshold=1.0e12, temporal_reduction="max", chunk_size=10
    )

    assert maximum.spatial_score[4, 4] > mean.spatial_score[4, 4]
    assert pt.equal(maximum.spatial_score, unchunked.spatial_score)


def test_curvilinear_coordinates_set_anisotropic_physical_patch():
    data = pt.zeros((9, 15, 2))
    i = pt.arange(9.0)[:, None]
    j = pt.arange(15.0)[None, :]
    x, y = pt.broadcast_tensors(2.0 * i, j)

    _, diagnostics = create_image_mask(data, x=x, y=y, physical_half_width=4.0)

    assert diagnostics.grid_spacing == pytest.approx((2.0, 1.0))
    assert diagnostics.patch_size == (5, 9)


def test_z_coordinate_accounts_for_surface_curvature():
    data = pt.zeros((11, 11))
    i = pt.arange(11.0)[:, None]
    j = pt.arange(11.0)[None, :]
    x, y = pt.broadcast_tensors(i, j)
    z = 2.0 * x

    _, planar = create_image_mask(data, x=x, y=y, physical_half_width=2.0)
    _, curved = create_image_mask(data, x=x, y=y, z=z, physical_half_width=2.0)

    assert planar.patch_size == (5, 5)
    assert curved.patch_size == (3, 5)
    assert curved.grid_spacing == pytest.approx((5.0**0.5, 1.0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lower_bound": 1.0, "upper_bound": 0.0}, "lower_bound"),
        ({"threshold": 0.0}, "threshold"),
        ({"patch_size": 4}, "patch dimensions"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"temporal_reduction": "median"}, "temporal_reduction"),
        ({"physical_half_width": 1.0}, "coordinates"),
    ],
)
def test_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        create_image_mask(pt.zeros((7, 7, 2)), **kwargs)


def test_invalid_coordinates():
    data = pt.zeros((7, 7))
    coordinate = pt.zeros((7, 7))

    with pytest.raises(ValueError, match="x and y"):
        create_image_mask(data, x=coordinate)
    with pytest.raises(ValueError, match="spatial data shape"):
        create_image_mask(data, x=coordinate[:-1], y=coordinate[:-1])
