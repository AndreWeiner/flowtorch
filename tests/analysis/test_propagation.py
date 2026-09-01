"""Tests for propagation quantities inferred from complex harmonic modes."""

import pytest
import torch as pt

from flowtorch.analysis import curvilinear_surface_phase_velocity


def _planar_grid(nx: int = 41, ny: int = 37) -> tuple[pt.Tensor, pt.Tensor]:
    first = pt.linspace(-1.0, 1.0, nx, dtype=pt.float64)
    second = pt.linspace(-0.8, 0.8, ny, dtype=pt.float64)
    return pt.meshgrid(first, second, indexing="ij")


def _traveling_mode(x: pt.Tensor, y: pt.Tensor, kx: float, ky: float) -> pt.Tensor:
    return pt.exp(-1j * (kx * x + ky * y))


def test_single_planar_mode_phase_velocity():
    x, y = _planar_grid()
    frequency = 3.0
    kx, ky = 2.0, -1.5
    mode = _traveling_mode(x, y, kx, ky)

    velocity = curvilinear_surface_phase_velocity(mode, frequency, x, y)

    expected = (
        2.0 * pt.pi * frequency * pt.tensor([kx, ky], dtype=x.dtype) / (kx**2 + ky**2)
    )
    assert velocity.shape == (*x.shape, 2)
    pt.testing.assert_close(
        velocity[2:-2, 2:-2],
        expected.expand_as(velocity[2:-2, 2:-2]),
        rtol=5.0e-3,
        atol=5.0e-3,
    )


def test_global_phase_does_not_change_velocity():
    x, y = _planar_grid()
    mode = _traveling_mode(x, y, 1.5, 0.5)
    shifted = mode * pt.exp(1j * pt.tensor(1.2, dtype=x.dtype))

    original_velocity = curvilinear_surface_phase_velocity(mode, 2.0, x, y)
    shifted_velocity = curvilinear_surface_phase_velocity(shifted, 2.0, x, y)

    pt.testing.assert_close(original_velocity, shifted_velocity)


def test_multiple_modes_use_one_frequency_per_mode():
    x, y = _planar_grid()
    modes = pt.stack(
        (
            _traveling_mode(x, y, 2.0, 1.0),
            _traveling_mode(x, y, -1.0, 2.5),
        ),
        dim=-1,
    )
    frequencies = pt.tensor([2.0, -4.0], dtype=x.dtype)

    velocity = curvilinear_surface_phase_velocity(modes, frequencies, x, y)

    assert velocity.shape == (*x.shape, 2, 2)
    for mode_index in range(2):
        expected = curvilinear_surface_phase_velocity(
            modes[..., mode_index],
            float(frequencies[mode_index]),
            x,
            y,
        )
        pt.testing.assert_close(velocity[..., mode_index, :], expected)


def test_amplitude_mask_is_independent_for_each_mode():
    x, y = _planar_grid()
    mode = _traveling_mode(x, y, 2.0, 1.0)
    modes = pt.stack((mode, 1.0e-6 * mode), dim=-1)
    frequencies = pt.tensor([2.0, 2.0], dtype=x.dtype)

    velocity = curvilinear_surface_phase_velocity(
        modes,
        frequencies,
        x,
        y,
        amplitude_threshold=0.5,
    )

    assert bool(pt.isfinite(velocity).all())
    pt.testing.assert_close(velocity[..., 0, :], velocity[..., 1, :])


def test_weak_amplitudes_and_small_wavenumbers_are_masked():
    x, y = _planar_grid()
    mode = _traveling_mode(x, y, 2.0, 1.0)
    mode[x.shape[0] // 2, y.shape[1] // 2] = 0.0

    velocity = curvilinear_surface_phase_velocity(
        mode, 2.0, x, y, amplitude_threshold=0.5
    )
    assert bool(pt.isnan(velocity[x.shape[0] // 2, y.shape[1] // 2]).all())

    masked = curvilinear_surface_phase_velocity(mode, 2.0, x, y, min_wavenumber=10.0)
    assert bool(pt.isnan(masked).all())


def test_embedded_surface_velocity_is_tangential():
    coordinate_u, coordinate_v = _planar_grid()
    x = coordinate_u
    y = coordinate_v
    z = coordinate_u
    frequency = 2.0
    wavenumber = 1.5
    mode = pt.exp(-1j * wavenumber * coordinate_u)

    velocity = curvilinear_surface_phase_velocity(mode, frequency, x, y, z)

    expected = (
        2.0 * pt.pi * frequency / wavenumber * pt.tensor([1.0, 0.0, 1.0], dtype=x.dtype)
    )
    assert velocity.shape == (*x.shape, 3)
    pt.testing.assert_close(
        velocity[2:-2, 2:-2],
        expected.expand_as(velocity[2:-2, 2:-2]),
        rtol=5.0e-3,
        atol=5.0e-3,
    )


def test_smoothing_processes_all_modes():
    x, y = _planar_grid()
    modes = pt.stack(
        (
            _traveling_mode(x, y, 2.0, 1.0),
            _traveling_mode(x, y, -1.0, 2.0),
        ),
        dim=-1,
    )

    velocity = curvilinear_surface_phase_velocity(
        modes,
        pt.tensor([2.0, 3.0], dtype=x.dtype),
        x,
        y,
        smoothing_sigma=0.5,
    )

    assert velocity.shape == (*x.shape, 2, 2)
    assert bool(pt.isfinite(velocity).all())


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        (2.0, "one-dimensional tensor"),
        (pt.tensor(2.0), "one-dimensional tensor"),
        (pt.tensor([2.0]), "exactly one value per mode"),
        (pt.tensor([2.0, 0.0]), "nonzero"),
        (pt.tensor([2.0, float("nan")]), "finite"),
    ],
)
def test_multiple_modes_require_one_valid_frequency_each(frequencies, message):
    x, y = _planar_grid()
    mode = _traveling_mode(x, y, 2.0, 1.0)
    modes = pt.stack((mode, mode), dim=-1)

    with pytest.raises(ValueError, match=message):
        curvilinear_surface_phase_velocity(modes, frequencies, x, y)


@pytest.mark.parametrize(
    ("modes", "frequencies", "message"),
    [
        (pt.ones((4, 5)), 2.0, "complex dtype"),
        (pt.ones((4, 5), dtype=pt.complex64), 0.0, "nonzero"),
        (pt.ones((4, 5), dtype=pt.complex64), pt.tensor([2.0]), "scalar"),
    ],
)
def test_single_mode_validation(modes, frequencies, message):
    x, y = pt.meshgrid(pt.arange(4.0), pt.arange(5.0), indexing="ij")

    with pytest.raises(ValueError, match=message):
        curvilinear_surface_phase_velocity(modes, frequencies, x, y)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("amplitude_threshold", -0.1, "amplitude_threshold"),
        ("amplitude_threshold", 1.1, "amplitude_threshold"),
        ("min_wavenumber", -1.0, "min_wavenumber"),
        ("smoothing_sigma", -1.0, "smoothing sigma"),
        ("smoothing_mode", "invalid", "smoothing_mode"),
        ("smoothing_truncate", 0.0, "smoothing_truncate"),
    ],
)
def test_phase_velocity_option_validation(keyword, value, message):
    x, y = _planar_grid()
    mode = _traveling_mode(x, y, 2.0, 1.0)

    with pytest.raises(ValueError, match=message):
        curvilinear_surface_phase_velocity(mode, 2.0, x, y, **{keyword: value})
