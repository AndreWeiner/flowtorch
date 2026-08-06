import pytest
import torch as pt

from flowtorch.visualization import line_integral_convolution


def _planar_grid(size=7):
    return pt.meshgrid(pt.arange(float(size)), pt.arange(float(size)), indexing="ij")


def test_lic_is_deterministic_with_seed():
    x, y = _planar_grid()
    vector = pt.stack((pt.ones_like(x), pt.zeros_like(y)), dim=-1)

    first = line_integral_convolution(vector, x, y, seed=13)
    second = line_integral_convolution(vector, x, y, seed=13)

    assert pt.equal(first, second)


def test_zero_vector_returns_explicit_texture_without_normalization():
    x, y = _planar_grid()
    vector = pt.zeros((*x.shape, 2))
    texture = (x + y) / 12.0

    lic = line_integral_convolution(vector, x, y, texture=texture, normalize=False)

    assert pt.equal(lic, texture)


def test_constant_vector_smooths_along_its_direction():
    x, y = _planar_grid()
    vector = pt.stack((pt.ones_like(x), pt.zeros_like(y)), dim=-1)
    texture = pt.zeros_like(x)
    texture[3, 3] = 1.0

    lic = line_integral_convolution(
        vector, x, y, steps=2, step_size=1.0, texture=texture, normalize=False
    )

    assert 0.0 < lic[3, 3] < 1.0
    assert lic[2, 3] > 0.0
    assert lic[4, 3] > 0.0
    assert lic[3, 2] == 0.0


def test_sequence_uses_shared_texture():
    x, y = _planar_grid()
    vector = pt.stack((pt.ones_like(x), pt.zeros_like(y)), dim=-1)
    sequence = pt.stack((vector, vector), dim=2)

    lic = line_integral_convolution(sequence, x, y, seed=4)

    assert lic.shape == (*x.shape, 2)
    assert pt.equal(lic[..., 0], lic[..., 1])


def test_pure_surface_normal_does_not_advect_texture():
    x, y = _planar_grid()
    z = x
    normal = pt.stack((-pt.ones_like(x), pt.zeros_like(x), pt.ones_like(x)), dim=-1)
    texture = (x + y) / 12.0

    lic = line_integral_convolution(normal, x, y, z, texture=texture, normalize=False)

    assert pt.allclose(lic, texture)


def test_surface_tangent_vector_follows_grid_direction():
    x, y = _planar_grid()
    z = x
    tangent = pt.stack((pt.ones_like(x), pt.zeros_like(x), pt.ones_like(x)), dim=-1)
    planar = pt.stack((pt.ones_like(x), pt.zeros_like(x)), dim=-1)
    texture = pt.zeros_like(x)
    texture[3, 3] = 1.0

    surface_lic = line_integral_convolution(
        tangent, x, y, z, steps=2, step_size=1.0, texture=texture, normalize=False
    )
    planar_lic = line_integral_convolution(
        planar, x, y, steps=2, step_size=1.0, texture=texture, normalize=False
    )

    assert pt.allclose(surface_lic, planar_lic)


def test_stretched_coordinates_preserve_streamline_direction():
    i, j = _planar_grid()
    x, y = 2.0 * i, 0.5 * j
    vector = pt.stack((pt.ones_like(x), pt.zeros_like(y)), dim=-1)
    texture = pt.zeros_like(x)
    texture[3, 3] = 1.0

    lic = line_integral_convolution(
        vector, x, y, steps=2, step_size=1.0, texture=texture, normalize=False
    )

    assert lic[2, 3] > 0.0
    assert lic[4, 3] > 0.0
    assert lic[3, 2] == 0.0


def test_lic_normalizes_finite_output():
    x, y = _planar_grid()
    vector = pt.zeros((*x.shape, 2))
    texture = x + y

    lic = line_integral_convolution(vector, x, y, texture=texture)

    assert pt.min(lic) == 0.0
    assert pt.max(lic) == 1.0


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("steps", 0),
        ("step_size", 0.0),
        ("metric_tolerance", -1.0),
    ],
)
def test_lic_rejects_invalid_options(keyword, value):
    x, y = _planar_grid()
    vector = pt.zeros((*x.shape, 2))
    with pytest.raises(ValueError):
        line_integral_convolution(vector, x, y, **{keyword: value})


def test_lic_rejects_texture_and_seed():
    x, y = _planar_grid()
    vector = pt.zeros((*x.shape, 2))
    with pytest.raises(ValueError):
        line_integral_convolution(vector, x, y, texture=x, seed=1)
