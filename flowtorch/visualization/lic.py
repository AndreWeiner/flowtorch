"""Line integral convolution on structured curvilinear grids."""

# standard library packages
from math import isfinite

# third party packages
import torch as pt
from torch.nn import functional as F


def line_integral_convolution(
    vector_field: pt.Tensor,
    x: pt.Tensor,
    y: pt.Tensor,
    z: pt.Tensor | None = None,
    steps: int = 30,
    step_size: float = 0.5,
    texture: pt.Tensor | None = None,
    seed: int | None = None,
    normalize: bool = True,
    metric_tolerance: float = 1.0e-12,
) -> pt.Tensor:
    """Compute line integral convolution on a curvilinear grid.

    The vector field is supplied in physical Cartesian components and is
    transformed to structured-grid index directions before streamline
    integration. A single planar vector field has shape ``(nx, ny, 2)``; a
    field tangent to an embedded surface has shape ``(nx, ny, 3)`` and is
    accompanied by ``z``. Sequences have shape
    ``(nx, ny, n_snapshots, n_components)`` and produce LIC images with shape
    ``(nx, ny, n_snapshots)``.

    Forward and backward streamlines are advanced with explicit Euler steps.
    ``step_size`` is measured in grid-index units, and texture samples along a
    streamline receive uniform weight. If no texture is supplied, a white-noise
    texture is generated. The same texture is shared by every snapshot to
    avoid artificial temporal flicker. A supplied texture may have shape
    ``(nx, ny)`` or, for a sequence, ``(nx, ny, n_snapshots)``.

    :param vector_field: physical vector field or vector-field sequence
    :type vector_field: pt.Tensor
    :param x: first Cartesian coordinate on the structured grid
    :type x: pt.Tensor
    :param y: second Cartesian coordinate on the structured grid
    :type y: pt.Tensor
    :param z: optional third Cartesian coordinate on an embedded surface
    :type z: pt.Tensor, optional
    :param steps: integration steps in each streamline direction, defaults to 30
    :type steps: int, optional
    :param step_size: integration step in grid-index units, defaults to 0.5
    :type step_size: float, optional
    :param texture: input texture; defaults to shared white noise
    :type texture: pt.Tensor, optional
    :param seed: local random seed; cannot be combined with ``texture``
    :type seed: int, optional
    :param normalize: normalize each finite LIC image to ``[0, 1]``
    :type normalize: bool, optional
    :param metric_tolerance: relative threshold for singular grid metrics
    :type metric_tolerance: float, optional
    :return: LIC image or LIC image sequence
    :rtype: pt.Tensor

    **Examples**

    .. code-block:: python

        vector = pt.stack((velocity_x, velocity_y), dim=-1)
        lic = line_integral_convolution(vector, x, y, seed=0)

        surface_vector = pt.stack((velocity_x, velocity_y, velocity_z), dim=-1)
        surface_lic = line_integral_convolution(
            surface_vector,
            x,
            y,
            z,
            steps=40,
            step_size=0.5,
            seed=0,
        )

    **Reference**

    B. Cabral and L. C. Leedom, "Imaging Vector Fields Using Line Integral
    Convolution," *Proceedings of SIGGRAPH 1993*, pp. 263--270, 1993.
    """
    coordinates, sequence, single_snapshot = _validate_inputs(
        vector_field,
        x,
        y,
        z,
        steps,
        step_size,
        texture,
        seed,
        metric_tolerance,
    )
    nx, ny, n_snapshots, _ = sequence.shape
    textures = _prepare_textures(
        texture,
        nx,
        ny,
        n_snapshots,
        vector_field,
        seed,
    )
    result = pt.empty(
        (nx, ny, n_snapshots),
        dtype=vector_field.dtype,
        device=vector_field.device,
    )
    for snapshot in range(n_snapshots):
        velocity_i, velocity_j = _grid_index_velocity(
            sequence[:, :, snapshot], coordinates, metric_tolerance
        )
        result[:, :, snapshot] = _convolve_texture(
            textures[:, :, snapshot],
            velocity_i,
            velocity_j,
            steps,
            step_size,
        )
        if normalize:
            result[:, :, snapshot] = _normalize_finite(result[:, :, snapshot])
    return result[:, :, 0] if single_snapshot else result


def _grid_index_velocity(
    vector_field: pt.Tensor,
    coordinates: tuple[pt.Tensor, ...],
    metric_tolerance: float,
) -> tuple[pt.Tensor, pt.Tensor]:
    position = pt.stack(coordinates, dim=-1)
    edge_order = 2 if min(position.shape[:2]) >= 3 else 1
    dr_di, dr_dj = pt.gradient(position, dim=(0, 1), edge_order=edge_order)
    metric_ii = pt.sum(dr_di * dr_di, dim=-1)
    metric_ij = pt.sum(dr_di * dr_dj, dim=-1)
    metric_jj = pt.sum(dr_dj * dr_dj, dim=-1)
    determinant = metric_ii * metric_jj - metric_ij.square()
    metric_scale = metric_ii * metric_jj
    invalid = determinant <= metric_tolerance * metric_scale
    invalid |= ~pt.isfinite(vector_field).all(dim=-1)
    safe_determinant = pt.where(invalid, pt.ones_like(determinant), determinant)

    projection_i = pt.sum(dr_di * vector_field, dim=-1)
    projection_j = pt.sum(dr_dj * vector_field, dim=-1)
    velocity_i = metric_jj * projection_i - metric_ij * projection_j
    velocity_i /= safe_determinant
    velocity_j = -metric_ij * projection_i + metric_ii * projection_j
    velocity_j /= safe_determinant
    velocity_i = velocity_i.masked_fill(invalid, float("nan"))
    velocity_j = velocity_j.masked_fill(invalid, float("nan"))
    return velocity_i, velocity_j


def _convolve_texture(
    texture: pt.Tensor,
    velocity_i: pt.Tensor,
    velocity_j: pt.Tensor,
    steps: int,
    step_size: float,
) -> pt.Tensor:
    nx, ny = texture.shape
    initial_i, initial_j = pt.meshgrid(
        pt.arange(nx, dtype=texture.dtype, device=texture.device),
        pt.arange(ny, dtype=texture.dtype, device=texture.device),
        indexing="ij",
    )
    finite_texture = pt.isfinite(texture)
    total = pt.where(finite_texture, texture, pt.zeros_like(texture))
    weight = finite_texture.to(dtype=texture.dtype)
    epsilon = pt.finfo(texture.dtype).eps

    for direction in (-1.0, 1.0):
        position_i = initial_i.clone()
        position_j = initial_j.clone()
        active = finite_texture.clone()
        for _ in range(steps):
            sampled_i = _sample_bilinear(velocity_i, position_i, position_j)
            sampled_j = _sample_bilinear(velocity_j, position_i, position_j)
            speed = pt.sqrt(sampled_i.square() + sampled_j.square())
            valid = active & pt.isfinite(speed) & (speed > epsilon)
            safe_speed = pt.where(valid, speed, pt.ones_like(speed))
            next_i = position_i + direction * step_size * sampled_i / safe_speed
            next_j = position_j + direction * step_size * sampled_j / safe_speed
            valid &= (
                (next_i >= 0.0)
                & (next_i <= nx - 1)
                & (next_j >= 0.0)
                & (next_j <= ny - 1)
            )
            sampled_texture = _sample_bilinear(texture, next_i, next_j)
            valid &= pt.isfinite(sampled_texture)
            total = total + pt.where(valid, sampled_texture, pt.zeros_like(total))
            weight = weight + valid.to(dtype=weight.dtype)
            position_i = pt.where(valid, next_i, position_i)
            position_j = pt.where(valid, next_j, position_j)
            active = valid
            if not active.any():
                break
    return pt.where(weight > 0.0, total / weight, pt.full_like(total, float("nan")))


def _sample_bilinear(
    field: pt.Tensor, position_i: pt.Tensor, position_j: pt.Tensor
) -> pt.Tensor:
    nx, ny = field.shape
    safe_i = pt.where(pt.isfinite(position_i), position_i, pt.zeros_like(position_i))
    safe_j = pt.where(pt.isfinite(position_j), position_j, pt.zeros_like(position_j))
    normalized_i = 2.0 * safe_i / max(nx - 1, 1) - 1.0
    normalized_j = 2.0 * safe_j / max(ny - 1, 1) - 1.0
    grid = pt.stack((normalized_j, normalized_i), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        field.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, 0]


def _prepare_textures(
    texture: pt.Tensor | None,
    nx: int,
    ny: int,
    n_snapshots: int,
    reference: pt.Tensor,
    seed: int | None,
) -> pt.Tensor:
    if texture is None:
        if seed is None:
            shared = pt.rand((nx, ny), dtype=reference.dtype, device=reference.device)
        else:
            generator = pt.Generator(device=reference.device)
            generator.manual_seed(seed)
            shared = pt.rand(
                (nx, ny),
                dtype=reference.dtype,
                device=reference.device,
                generator=generator,
            )
        return shared.unsqueeze(-1).expand(nx, ny, n_snapshots)
    if texture.ndim == 2:
        return texture.unsqueeze(-1).expand(nx, ny, n_snapshots)
    return texture


def _normalize_finite(image: pt.Tensor) -> pt.Tensor:
    finite = pt.isfinite(image)
    if not finite.any():
        return image
    minimum = pt.min(image[finite])
    maximum = pt.max(image[finite])
    if maximum <= minimum:
        return image
    return pt.where(finite, (image - minimum) / (maximum - minimum), image)


def _validate_inputs(
    vector_field: pt.Tensor,
    x: pt.Tensor,
    y: pt.Tensor,
    z: pt.Tensor | None,
    steps: int,
    step_size: float,
    texture: pt.Tensor | None,
    seed: int | None,
    metric_tolerance: float,
) -> tuple[tuple[pt.Tensor, ...], pt.Tensor, bool]:
    coordinates = (x, y) if z is None else (x, y, z)
    if any(coordinate.ndim != 2 for coordinate in coordinates):
        raise ValueError("coordinates must be two-dimensional")
    if any(coordinate.shape != x.shape for coordinate in coordinates[1:]):
        raise ValueError("coordinates must have identical shapes")
    if min(x.shape) < 2:
        raise ValueError("each grid dimension must contain at least two points")
    if vector_field.ndim not in (3, 4):
        raise ValueError(
            "vector_field must have shape (nx, ny, n_components) or "
            "(nx, ny, n_snapshots, n_components)"
        )
    if vector_field.shape[:2] != x.shape:
        raise ValueError("vector-field and coordinate spatial shapes must match")
    if vector_field.shape[-1] != len(coordinates):
        raise ValueError("vector-field components must match coordinate dimensions")
    if not vector_field.is_floating_point():
        raise ValueError("vector_field must have a floating-point dtype")
    for coordinate in coordinates:
        if (
            coordinate.dtype != vector_field.dtype
            or coordinate.device != vector_field.device
        ):
            raise ValueError("vector field and coordinates must share dtype and device")
        if not pt.isfinite(coordinate).all():
            raise ValueError("coordinates must contain only finite values")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be at least 1")
    if not isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    if not isfinite(metric_tolerance) or metric_tolerance < 0.0:
        raise ValueError("metric_tolerance must be finite and non-negative")
    if texture is not None and seed is not None:
        raise ValueError("texture and seed cannot be supplied together")

    single_snapshot = vector_field.ndim == 3
    sequence = vector_field.unsqueeze(2) if single_snapshot else vector_field
    n_snapshots = sequence.shape[2]
    if texture is not None:
        if texture.dtype != vector_field.dtype or texture.device != vector_field.device:
            raise ValueError("texture and vector field must share dtype and device")
        valid_texture_shape = texture.shape == x.shape or texture.shape == (
            *x.shape,
            n_snapshots,
        )
        if not valid_texture_shape:
            raise ValueError("texture shape must match the grid or vector sequence")
        if single_snapshot and texture.ndim != 2:
            raise ValueError("a single vector field requires a two-dimensional texture")
    return coordinates, sequence, single_snapshot
