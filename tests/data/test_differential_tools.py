import pytest
import torch as pt

from flowtorch.data import (
    curvilinear_gradient,
    curvilinear_hessian,
    curvilinear_jacobian,
)


def test_curvilinear_gradient_on_stretched_planar_grid():
    i, j = pt.meshgrid(pt.arange(5.0), pt.arange(6.0), indexing="ij")
    x = (i + 1.0) ** 2 + 0.2 * j
    y = 0.1 * i + (j + 1.0) ** 2
    field = 3.0 * x - 2.0 * y

    gradient = curvilinear_gradient(field, x, y)

    expected = pt.tensor([3.0, -2.0]).expand_as(gradient)
    assert gradient.shape == (*field.shape, 2)
    assert pt.allclose(gradient, expected, atol=1.0e-5)


def test_curvilinear_gradient_processes_snapshots_independently():
    x, y = pt.meshgrid(pt.arange(5.0), pt.arange(6.0), indexing="ij")
    field = pt.stack((2.0 * x, -3.0 * y), dim=-1)

    gradient = curvilinear_gradient(field, x, y)

    assert gradient.shape == (*field.shape, 2)
    assert pt.allclose(gradient[..., 0, :], pt.tensor([2.0, 0.0]))
    assert pt.allclose(gradient[..., 1, :], pt.tensor([0.0, -3.0]))


def test_curvilinear_gradient_on_embedded_plane():
    x, y = pt.meshgrid(pt.arange(5.0), pt.arange(6.0), indexing="ij")
    z = x
    field = x

    gradient = curvilinear_gradient(field, x, y, z)

    expected = pt.tensor([0.5, 0.0, 0.5]).expand_as(gradient)
    assert gradient.shape == (*field.shape, 3)
    assert pt.allclose(gradient, expected)


def test_curvilinear_gradient_marks_singular_grid_as_nan():
    x = pt.zeros((3, 3))
    y = pt.zeros((3, 3))
    field = pt.ones((3, 3))

    gradient = curvilinear_gradient(field, x, y)

    assert pt.isnan(gradient).all()


def test_gaussian_smoothing_reduces_gradient_peak():
    x, y = pt.meshgrid(pt.arange(9.0), pt.arange(9.0), indexing="ij")
    field = pt.zeros_like(x)
    field[4, 4] = 1.0

    raw = curvilinear_gradient(field, x, y)
    smooth = curvilinear_gradient(field, x, y, smoothing_sigma=1.0)

    assert pt.max(pt.linalg.vector_norm(smooth, dim=-1)) < pt.max(
        pt.linalg.vector_norm(raw, dim=-1)
    )


def test_gradient_outlier_replacement_is_spatial():
    x, y = pt.meshgrid(pt.arange(9.0), pt.arange(9.0), indexing="ij")
    field = x.clone()
    field[4, 4] = 100.0

    raw = curvilinear_gradient(field, x, y)
    clean = curvilinear_gradient(field, x, y, outlier_threshold=3.5)

    expected = pt.tensor([1.0, 0.0]).expand_as(raw)
    assert pt.max(pt.abs(raw - expected)) > 1.0
    assert pt.allclose(clean, expected)


def test_curvilinear_jacobian_on_planar_grid():
    x, y = pt.meshgrid(pt.arange(5.0), pt.arange(6.0), indexing="ij")
    field = pt.stack((2.0 * x + 3.0 * y, -x + 4.0 * y), dim=-1)

    jacobian = curvilinear_jacobian(field, x, y)

    expected = pt.tensor([[2.0, 3.0], [-1.0, 4.0]]).expand_as(jacobian)
    assert jacobian.shape == (*x.shape, 2, 2)
    assert pt.allclose(jacobian, expected)


def test_curvilinear_jacobian_processes_snapshot_sequence():
    x, y = pt.meshgrid(pt.arange(5.0), pt.arange(6.0), indexing="ij")
    first = pt.stack((x, y), dim=-1)
    second = pt.stack((2.0 * x, 3.0 * y), dim=-1)
    field = pt.stack((first, second), dim=2)

    jacobian = curvilinear_jacobian(field, x, y)

    assert jacobian.shape == (*x.shape, 2, 2, 2)
    assert pt.allclose(jacobian[..., 0, :, :], pt.eye(2))
    assert pt.allclose(jacobian[..., 1, :, :], pt.diag(pt.tensor([2.0, 3.0])))


def test_curvilinear_hessian_on_planar_grid():
    x, y = pt.meshgrid(pt.arange(6.0), pt.arange(7.0), indexing="ij")
    field = x**2 + 3.0 * x * y + 2.0 * y**2

    hessian = curvilinear_hessian(field, x, y)

    expected = pt.tensor([[2.0, 3.0], [3.0, 4.0]]).expand_as(hessian)
    assert hessian.shape == (*x.shape, 2, 2)
    assert pt.allclose(hessian, expected)


def test_curvilinear_hessian_processes_snapshot_sequence():
    x, y = pt.meshgrid(pt.arange(6.0), pt.arange(7.0), indexing="ij")
    field = pt.stack((x**2, y**2), dim=-1)

    hessian = curvilinear_hessian(field, x, y)

    assert hessian.shape == (*field.shape, 2, 2)
    assert pt.allclose(hessian[..., 0, :, :], pt.diag(pt.tensor([2.0, 0.0])))
    assert pt.allclose(hessian[..., 1, :, :], pt.diag(pt.tensor([0.0, 2.0])))


def test_curvilinear_hessian_on_embedded_plane():
    x, y = pt.meshgrid(pt.arange(6.0), pt.arange(7.0), indexing="ij")
    z = x
    field = x**2

    hessian = curvilinear_hessian(field, x, y, z)

    expected = pt.tensor([[0.5, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]).expand_as(
        hessian
    )
    assert hessian.shape == (*field.shape, 3, 3)
    assert pt.allclose(hessian, expected)


def test_hessian_accepts_different_processing_for_each_derivative():
    x, y = pt.meshgrid(pt.arange(9.0), pt.arange(9.0), indexing="ij")
    field = x**2 + y**2

    hessian = curvilinear_hessian(
        field,
        x,
        y,
        first_smoothing_sigma=0.5,
        second_smoothing_sigma=1.0,
        first_outlier_threshold=3.5,
        second_outlier_threshold=4.0,
    )

    assert hessian.shape == (*field.shape, 2, 2)
    assert pt.isfinite(hessian).all()


def test_curvilinear_jacobian_rejects_scalar_field():
    x, y = pt.meshgrid(pt.arange(3.0), pt.arange(3.0), indexing="ij")
    with pytest.raises(ValueError):
        curvilinear_jacobian(x + y, x, y)


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("smoothing_sigma", -1.0),
        ("smoothing_sigma", (1.0, -1.0)),
        ("smoothing_sigma", (1.0,)),
        ("smoothing_mode", "invalid"),
        ("smoothing_truncate", 0.0),
        ("edge_order", 3),
        ("metric_tolerance", -1.0),
        ("outlier_threshold", 0.0),
        ("outlier_window_size", 2),
    ],
)
def test_curvilinear_gradient_rejects_invalid_options(keyword, value):
    x, y = pt.meshgrid(pt.arange(3.0), pt.arange(3.0), indexing="ij")
    with pytest.raises(ValueError):
        curvilinear_gradient(x + y, x, y, **{keyword: value})
