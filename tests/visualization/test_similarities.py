import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as pt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flowtorch.analysis import AMSPOD, PODSubspaceDependencyResult
from flowtorch.visualization import (
    plot_mode_similarity,
    plot_pod_subspace_data_dependency,
    plot_pod_subspace_data_dependency_ranks,
)


def _pod_dependency_result():
    return PODSubspaceDependencyResult(
        similarity=pt.tensor(
            [
                [[0.7, 0.6], [0.5, 0.4]],
                [[1.0, 1.0], [0.9, 0.8]],
            ]
        ),
        ranks=pt.tensor([1, 3]),
        sequence_fractions=pt.tensor([0.5, 1.0]),
        snapshot_strides=pt.tensor([1, 2]),
        n_snapshots=pt.tensor([[50, 25], [100, 50]]),
        optimal_ranks=pt.tensor([[2, 2], [3, 2]]),
        reference_optimal_rank=3,
    )


def test_plot_pod_subspace_data_dependency():
    result = _pod_dependency_result()

    figure, axes = plot_pod_subspace_data_dependency(result, rank=3, annotate=True)

    plotted = axes.images[0].get_array()
    assert np.allclose(plotted, result.similarity[:, :, 1].T.numpy())
    assert axes.get_xlabel() == r"sequence length $[\%]$"
    assert axes.get_ylabel() == "snapshot stride"
    assert len(axes.texts) == 4
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == r"$S_{3}$"
    plt.close(figure)


def test_plot_pod_subspace_data_dependency_rank_curves():
    result = _pod_dependency_result()

    figure, axes = plot_pod_subspace_data_dependency_ranks(
        result, sequence_fraction=1.0
    )

    assert len(axes.lines) == 2
    assert np.array_equal(axes.lines[0].get_xdata(), result.ranks.numpy())
    assert np.allclose(axes.lines[1].get_ydata(), result.similarity[1, 1].numpy())
    assert axes.get_ylim() == pytest.approx((0.0, 1.02))
    assert axes.get_legend() is not None
    plt.close(figure)


def test_plot_pod_subspace_data_dependency_validates_selection():
    result = _pod_dependency_result()

    with pytest.raises(ValueError, match="not present"):
        plot_pod_subspace_data_dependency(result, rank=2)
    with pytest.raises(ValueError, match="not present"):
        plot_pod_subspace_data_dependency_ranks(result, sequence_fraction=0.6)


def test_plot_mode_similarity_masks_upper_triangle_automatically():
    frequency = pt.tensor([0.0, 1.0, 2.0])
    similarity = pt.tensor([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])

    figure, axes = plot_mode_similarity(frequency, frequency, similarity)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    mask = np.ma.getmaskarray(axes.collections[0].get_array()).reshape(3, 3)
    assert np.array_equal(mask, np.triu(np.ones((3, 3), dtype=bool), k=1))
    assert axes.get_aspect() == pytest.approx(1.0)
    assert axes.get_xlabel() == r"$f_1\;[\mathrm{Hz}]$"
    assert axes.get_ylabel() == r"$f_2\;[\mathrm{Hz}]$"
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == r"$\rho$"
    plt.close(figure)


def test_plot_mode_similarity_keeps_nonsymmetric_matrix_full():
    frequency = pt.tensor([0.0, 1.0, 2.0])
    similarity = pt.tensor([[1.0, 0.8, 0.3], [0.2, 1.0, float("nan")], [0.4, 0.5, 1.0]])

    figure, axes = plot_mode_similarity(
        frequency, frequency, similarity, show_colorbar=False
    )

    mask = np.ma.getmaskarray(axes.collections[0].get_array()).reshape(3, 3)
    assert mask.sum() == 1
    assert mask[2, 1]
    assert axes.get_aspect() == "auto"
    assert len(figure.axes) == 1
    plt.close(figure)


def test_plot_mode_similarity_sorts_wrapped_frequency():
    frequency = pt.tensor([0.0, 1.0, -2.0, -1.0])
    similarity = pt.arange(16, dtype=pt.float32).reshape(4, 4)
    similarity = (similarity + similarity.T) / similarity.max() / 2.0
    similarity.fill_diagonal_(1.0)
    order = pt.argsort(frequency)
    expected = similarity[order][:, order].T.numpy()

    figure, axes = plot_mode_similarity(frequency, frequency, similarity)

    plotted = axes.collections[0].get_array()
    values = np.ma.getdata(plotted).reshape(4, 4)
    mask = np.ma.getmaskarray(plotted).reshape(4, 4)
    assert np.allclose(values[~mask], expected[~mask])
    plt.close(figure)


def test_plot_mode_similarity_validates_lower_triangle():
    frequency = pt.tensor([0.0, 1.0])
    similarity = pt.tensor([[1.0, 0.2], [0.8, 1.0]])

    with pytest.raises(ValueError, match="requires matching coordinates"):
        plot_mode_similarity(frequency, frequency, similarity, triangle="lower")
    with pytest.raises(ValueError, match="triangle"):
        plot_mode_similarity(frequency, frequency, similarity, triangle="invalid")


def test_plot_mode_similarity_strouhal_and_existing_axes():
    frequency = pt.tensor([0.0, 1.0])
    similarity = pt.eye(2)
    figure, supplied_axes = plt.subplots()

    returned_figure, returned_axes = plot_mode_similarity(
        frequency,
        frequency,
        similarity,
        reference_timescale=2.0,
        triangle="full",
        ax=supplied_axes,
        show_colorbar=False,
    )

    assert returned_figure is figure
    assert returned_axes is supplied_axes
    assert supplied_axes.get_xlabel() == r"$St_1$"
    assert supplied_axes.get_ylabel() == r"$St_2$"
    plt.close(figure)


def test_AMSPOD_show_mode_similarity():
    spod = AMSPOD.__new__(AMSPOD)
    spod._frequency = pt.tensor([0.0, 1.0, 2.0])
    spod._modes = pt.rand((3, 4, 1), dtype=pt.complex64)
    spod._sqrt_weight = pt.ones((4, 1))
    spod._adaptive = False

    figure, axes = spod.show_mode_similarity()

    mask = np.ma.getmaskarray(axes.collections[0].get_array()).reshape(3, 3)
    assert np.array_equal(mask, np.triu(np.ones((3, 3), dtype=bool), k=1))
    plt.close(figure)
