"""Tools for creating masks used to select or exclude field data."""

# standard library packages
from pathlib import Path
from typing import List, Sequence, Tuple
import logging
import time

# third party packages
import torch as pt

logger = logging.getLogger(__name__)


def mask_box(vertices: pt.Tensor, lower: List[float], upper: List[float]) -> pt.Tensor:
    """Create a boolean mask to select all vertices in a box.

    This function may be used in conjunction with torch.masked_select to select
    all field values in a box, e.g., when building data matrices.

    :param vertices: tensor of vertices, where each column corresponds to a coordinate
    :type vertices: pt.Tensor
    :param lower: lower bounds of box; one value for each coordinate must be given
    :type lower: List[float]
    :param upper: upper bounds of box; one value for each coordinate must be given
    :type upper: List[float]
    :return: boolean mask that's *True* for every vertex inside the box
    :rtype: pt.Tensor

    """
    assert (
        len(vertices.shape) < 3
    ), "The vertices tensor cannot have more than two axes."
    dim_message = "Exactly one lower and upper bound must be given for each coordinate."
    if len(vertices.shape) == 1:
        assert len(lower) == len(upper) == 1, dim_message
        return pt.logical_and(
            pt.where(vertices >= lower[0], True, False),
            pt.where(vertices <= upper[0], True, False),
        )
    else:
        assert vertices.shape[1] == len(lower) == len(upper), dim_message
        return pt.all(
            pt.logical_and(
                pt.where(vertices >= pt.tensor(lower), True, False),
                pt.where(vertices <= pt.tensor(upper), True, False),
            ),
            dim=1,
        )


def mask_sphere(vertices: pt.Tensor, center: List[float], radius: float) -> pt.Tensor:
    """Create a boolean mask to select all vertices in a sphere.

    This function may be used in conjunction with torch.masked_select to select
    all field values within a sphere, e.g., when building data matrices.

    :param vertices: tensor of vertices, where each column corresponds to a coordinate
    :type vertices: pt.Tensor
    :param center: the sphere's center
    :type center: List[float]
    :param radius: the sphere's radius
    :type radius: float
    :return: boolean mask that's *True* for every vertex inside the sphere
    :rtype: pt.Tensor
    """
    center_tensor = pt.tensor(center).type(vertices.dtype)
    assert (
        len(vertices.shape) < 3
    ), "The vertices tensor cannot have more than two axes."
    if len(vertices.shape) == 1:
        assert len(center_tensor) == 1
        radii = pt.abs(vertices - center_tensor)
    else:
        assert (
            vertices.shape[1] == center_tensor.shape[0]
        ), "Missmatch between number of vertices and center coordinates."
        radii = pt.linalg.norm(vertices - center_tensor, dim=1)
    return pt.where(radii <= radius, True, False)


def mask_polygon(shape: Sequence[int], vertices: pt.Tensor) -> pt.Tensor:
    """Create a boolean mask for array indices enclosed by a polygon.

    The polygon coordinates follow image-index order: the first coordinate
    addresses the first array dimension and the second coordinate addresses
    the second array dimension.

    :param shape: shape of the two-dimensional field
    :type shape: Sequence[int]
    :param vertices: polygon vertices with shape ``(n_vertices, 2)``
    :type vertices: pt.Tensor
    :return: mask that is ``True`` for indices inside the polygon
    :rtype: pt.Tensor
    """
    if len(shape) != 2:
        raise ValueError("shape must describe a two-dimensional field")
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (n_vertices, 2)")
    if vertices.shape[0] < 3:
        return pt.zeros(tuple(shape), dtype=pt.bool, device=vertices.device)

    # Matplotlib is optional and is only needed by the polygon-based tools.
    from matplotlib.path import Path as MatplotlibPath

    i, j = pt.meshgrid(pt.arange(shape[0]), pt.arange(shape[1]), indexing="ij")
    points = pt.stack((i.flatten(), j.flatten()), dim=1).numpy()
    polygon = MatplotlibPath(vertices.detach().cpu().numpy())
    mask = pt.from_numpy(polygon.contains_points(points).reshape(tuple(shape)))
    return mask.to(vertices.device)


def mask_image_interactive(
    image: pt.Tensor,
    initial_mask: pt.Tensor | None = None,
    output_path: str | Path | None = None,
    percentile: float = 95.0,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (7.0, 5.0),
    selection_pad: float = 0.05,
    view_update_interval: float = 0.05,
) -> pt.Tensor:
    """Interactively create a binary keep mask for a two-dimensional image.

    Draw a polygon and press ``a`` to exclude its points. Press ``u`` to undo,
    ``r`` to reset, ``i`` to invert, ``0`` to reset the view, ``s`` to save,
    or ``q`` to close the window. The returned and saved masks use ``True`` for
    retained points and ``False`` for excluded points.

    :param image: two-dimensional image on which to base the selection
    :type image: pt.Tensor
    :param initial_mask: initial keep mask; defaults to an all-True mask
    :type initial_mask: pt.Tensor, optional
    :param output_path: optional path at which ``s`` saves the mask as ``.pt``
    :type output_path: str or pathlib.Path, optional
    :param percentile: percentile used as the upper color limit
    :type percentile: float, optional
    :param cmap: Matplotlib colormap name, defaults to ``"viridis"``
    :type cmap: str, optional
    :param figsize: Matplotlib figure size, defaults to ``(7.0, 5.0)``
    :type figsize: Tuple[float, float], optional
    :param selection_pad: axes padding as a fraction of the image size
    :type selection_pad: float, optional
    :param view_update_interval: minimum delay between pan/zoom redraws
    :type view_update_interval: float, optional
    :return: the edited boolean keep mask
    :rtype: pt.Tensor
    """
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in the interval (0, 100]")
    if selection_pad < 0.0:
        raise ValueError("selection_pad must be non-negative")
    if view_update_interval < 0.0:
        raise ValueError("view_update_interval must be non-negative")

    if initial_mask is None:
        keep_mask = pt.ones_like(image, dtype=pt.bool)
    else:
        if initial_mask.shape != image.shape:
            raise ValueError("initial_mask must have the same shape as image")
        keep_mask = initial_mask.to(dtype=pt.bool).clone()

    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    image_values = image.detach().cpu().numpy()
    vmax = float(pt.nanquantile(image, percentile / 100.0))
    tool_keys = {"0", "a", "i", "r", "s", "u"}
    rc = {
        key: [value for value in plt.rcParams[key] if value not in tool_keys]
        for key in plt.rcParams
        if key.startswith("keymap.")
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)
        image_artist = ax.imshow(
            image_values.T,
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmax=vmax,
            interpolation="nearest",
            resample=False,
        )
        overlay = ax.imshow(
            (~keep_mask.detach().cpu()).T,
            origin="lower",
            aspect="equal",
            cmap="gray_r",
            alpha=0.35,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            resample=False,
        )
        fig.colorbar(image_artist, ax=ax)
        ax.set_xlabel("first array index")
        ax.set_ylabel("second array index")

        nx, ny = image.shape
        pad_x = max(1.0, selection_pad * nx)
        pad_y = max(1.0, selection_pad * ny)
        initial_xlim = (-0.5 - pad_x, nx - 0.5 + pad_x)
        initial_ylim = (-0.5 - pad_y, ny - 0.5 + pad_y)
        ax.set_xlim(initial_xlim)
        ax.set_ylim(initial_ylim)

        selected_vertices: list[tuple[float, float]] = []
        history: list[pt.Tensor] = []
        pan_start = None
        last_view_draw = 0.0

        def set_title(message: str | None = None) -> None:
            prefix = "a: exclude | u: undo | r: reset | i: invert | s: save"
            ax.set_title(prefix if message is None else f"{prefix}\n{message}")

        def update_overlay() -> None:
            overlay.set_data((~keep_mask.detach().cpu()).T)
            fig.canvas.draw_idle()

        def clear_selection() -> None:
            selected_vertices.clear()
            selector.clear()

        def on_select(vertices) -> None:
            selected_vertices[:] = vertices

        def request_view_draw(force: bool = False) -> None:
            nonlocal last_view_draw
            now = time.monotonic()
            if force or now - last_view_draw >= view_update_interval:
                last_view_draw = now
                fig.canvas.draw_idle()

        def on_key(event) -> None:
            if event.key == "a" and len(selected_vertices) >= 3:
                region = mask_polygon(image.shape, pt.tensor(selected_vertices))
                if region.any():
                    history.append(keep_mask.clone())
                    keep_mask[region.to(keep_mask.device)] = False
                    clear_selection()
                    set_title(f"Excluded {int(region.sum())} points")
                    update_overlay()
            elif event.key == "u" and history:
                keep_mask.copy_(history.pop())
                clear_selection()
                set_title("Undid last change")
                update_overlay()
            elif event.key == "r":
                history.append(keep_mask.clone())
                keep_mask.fill_(True)
                clear_selection()
                set_title("Mask reset")
                update_overlay()
            elif event.key == "i":
                history.append(keep_mask.clone())
                keep_mask.logical_not_()
                set_title("Mask inverted")
                update_overlay()
            elif event.key == "0":
                ax.set_xlim(initial_xlim)
                ax.set_ylim(initial_ylim)
                set_title("View reset")
                fig.canvas.draw_idle()
            elif event.key == "s":
                if output_path is None:
                    set_title("No output path configured")
                else:
                    path = Path(output_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pt.save(keep_mask.cpu(), path)
                    set_title(f"Saved keep mask to {path}")
                    logger.info("Saved keep mask to %s", path)
                fig.canvas.draw_idle()
            elif event.key == "q":
                plt.close(fig)

        def on_scroll(event) -> None:
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            scale = 1.0 / 1.25 if event.button == "up" else 1.25
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            width = (x1 - x0) * scale
            height = (y1 - y0) * scale
            relx = (event.xdata - x0) / (x1 - x0)
            rely = (event.ydata - y0) / (y1 - y0)
            ax.set_xlim(event.xdata - relx * width, event.xdata + (1 - relx) * width)
            ax.set_ylim(event.ydata - rely * height, event.ydata + (1 - rely) * height)
            request_view_draw()

        def on_button_press(event) -> None:
            nonlocal pan_start
            if event.inaxes == ax and event.button in (2, 3):
                pan_start = (event.x, event.y, ax.get_xlim(), ax.get_ylim())

        def on_button_release(event) -> None:
            nonlocal pan_start
            if event.button in (2, 3):
                pan_start = None
                request_view_draw(force=True)

        def on_motion(event) -> None:
            if pan_start is None:
                return
            x_start, y_start, xlim, ylim = pan_start
            dx = (event.x - x_start) / ax.bbox.width * (xlim[1] - xlim[0])
            dy = (event.y - y_start) / ax.bbox.height * (ylim[1] - ylim[0])
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            request_view_draw()

        selector = PolygonSelector(ax, on_select, useblit=True)
        set_title()
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        fig.canvas.mpl_connect("button_press_event", on_button_press)
        fig.canvas.mpl_connect("button_release_event", on_button_release)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        plt.show()

    return keep_mask


def mask_psp_interactive(
    path: str | Path,
    output_path: str | Path | None = None,
    n_snapshots: int | None = None,
    statistic: str = "std",
    **selection_options,
) -> tuple[pt.Tensor, pt.Tensor]:
    """Create a keep mask from a temporal statistic of PSP data.

    The structure of the PSP data in the HDF5 container is expected
    the follow the DLR (German Aerospace Center) practice.

    **Examples**

    .. code-block:: python

        mask, statistic = mask_psp_interactive("0226.hdf5")

        mask, statistic = mask_psp_interactive(
            "0226.hdf5",
            output_path="0226_std_mask.pt",
            n_snapshots=500,
            statistic="std",
            percentile=99.0,
        )

    :param path: path to the PSP HDF5 dataset
    :type path: str or pathlib.Path
    :param output_path: optional path at which ``s`` saves the mask
    :type output_path: str or pathlib.Path, optional
    :param n_snapshots: number of snapshots used to compute the statistic
    :type n_snapshots: int, optional
    :param statistic: temporal statistic, either ``"std"`` or ``"max"``
    :type statistic: str, optional
    :param selection_options: additional options for :func:`mask_image_interactive`
    :return: keep mask and displayed statistic image
    :rtype: tuple[pt.Tensor, pt.Tensor]
    """
    from .psp_dataloader import PSPDataloader

    loader = PSPDataloader(str(path))
    times = loader.write_times
    if n_snapshots is not None:
        if n_snapshots < 1:
            raise ValueError("n_snapshots must be positive")
        times = times[:n_snapshots]
    logger.info("Loading %d PSP snapshots from %s", len(times), path)
    fields = loader.load_snapshot("Cp", times)

    statistic = statistic.lower()
    if statistic == "std":
        statistic_image = pt.std(fields, dim=-1, unbiased=False)
    elif statistic == "max":
        statistic_image = pt.max(fields, dim=-1).values
    else:
        raise ValueError('statistic must be either "std" or "max"')

    if output_path is None:
        output_path = Path(path).with_name(f"{Path(path).stem}_{statistic}_mask.pt")
    keep_mask = mask_image_interactive(
        statistic_image,
        output_path=output_path,
        **selection_options,
    )
    return keep_mask, statistic_image
