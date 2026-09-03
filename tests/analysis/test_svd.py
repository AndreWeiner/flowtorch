"""Unittests for the SVD class."""

# third party libraries
import pytest
import torch as pt

# flowtorch packages
from flowtorch.analysis.svd import (
    SVD,
    pod_subspace_data_dependency,
    subspace_similarity,
)
from flowtorch.analysis.state_vector import (
    FieldSpec,
    StateVectorLayout,
    StateVectorSource,
)


class _MatrixSource(StateVectorSource):
    def __init__(self, data, weight=None):
        self.data = data
        self._layout = StateVectorLayout((FieldSpec("q"),), (data.shape[0],))
        self._weight = weight
        self.calls = []

    @property
    def n_snapshots(self):
        return self.data.shape[1]

    @property
    def layout(self):
        return self._layout

    def read(self, spatial_slice, snapshot_slice):
        self.calls.append((spatial_slice, snapshot_slice))
        return self.data[spatial_slice, snapshot_slice]

    def read_weight(self, spatial_slice):
        return None if self._weight is None else self._weight[spatial_slice]


def test_subspace_similarity_is_basis_invariant():
    first = pt.eye(3)[:, :2]
    rotation = pt.tensor([[2.0**-0.5, -(2.0**-0.5)], [2.0**-0.5, 2.0**-0.5]])
    second = first @ rotation

    similarity = subspace_similarity(first, second, ranks=(1, 2))

    pt.testing.assert_close(similarity, pt.tensor([0.5, 1.0]))


def test_subspace_similarity_measures_partial_overlap():
    first = pt.eye(3)[:, :2]
    second = pt.eye(3)[:, (0, 2)]

    similarity = subspace_similarity(first, second, ranks=2)

    pt.testing.assert_close(similarity, pt.tensor([0.5]))


def test_subspace_similarity_supports_weights_and_complex_modes():
    phase = pt.exp(1j * pt.tensor(0.7))
    first = pt.tensor([[1.0], [0.0]], dtype=pt.complex64)
    second = pt.tensor([[1.0], [1.0]], dtype=pt.complex64) * phase

    unweighted = subspace_similarity(first, second, ranks=1)
    weighted = subspace_similarity(first, second, ranks=1, weight=pt.tensor([4.0, 1.0]))

    pt.testing.assert_close(unweighted, pt.tensor([0.5]))
    pt.testing.assert_close(weighted, pt.tensor([0.8]))


@pytest.mark.parametrize(
    ("first", "second", "ranks", "weight", "message"),
    [
        (pt.ones(3), pt.ones((3, 1)), 1, None, "two-dimensional"),
        (pt.ones((3, 1)), pt.ones((2, 1)), 1, None, "state dimension"),
        (pt.ones((3, 2)), pt.ones((3, 2)), (2, 1), None, "increasing"),
        (pt.ones((3, 2)), pt.ones((3, 2)), 3, None, "available rank"),
        (pt.eye(2), pt.eye(2), 1, pt.tensor([1.0, 0.0]), "positive"),
    ],
)
def test_subspace_similarity_validates_inputs(first, second, ranks, weight, message):
    with pytest.raises(ValueError, match=message):
        subspace_similarity(first, second, ranks=ranks, weight=weight)


def _pod_dependency_data():
    time = pt.linspace(0.0, 4.0 * pt.pi, 40)
    coefficients = pt.stack(
        (pt.sin(time), 0.6 * pt.cos(2.0 * time), 0.2 * pt.sin(5.0 * time))
    )
    modes = pt.eye(8)[:, :3]
    return modes @ coefficients


def test_pod_subspace_data_dependency_sweep():
    data = _pod_dependency_data()

    result = pod_subspace_data_dependency(
        data,
        ranks=(1, 2),
        sequence_fractions=(0.5, 1.0),
        snapshot_strides=(1, 2),
        mode="svd",
    )

    assert result.similarity.shape == (2, 2, 2)
    assert result.optimal_ranks.shape == (2, 2)
    assert result.reference_optimal_rank >= 0
    pt.testing.assert_close(result.similarity[1, 0], pt.ones(2))
    assert pt.equal(result.ranks, pt.tensor([1, 2]))
    assert pt.equal(result.n_snapshots, pt.tensor([[20, 10], [40, 20]]))


def test_pod_subspace_data_dependency_uses_reference_automatic_rank():
    data = _pod_dependency_data()
    reference = SVD(data - data.mean(dim=1, keepdim=True), mode="svd")

    result = pod_subspace_data_dependency(
        data,
        sequence_fractions=(1.0,),
        snapshot_strides=(1,),
        mode="svd",
    )

    assert result.reference_optimal_rank == reference.opt_rank
    assert result.ranks[-1] == reference.rank
    pt.testing.assert_close(result.similarity, pt.ones_like(result.similarity))


def test_pod_subspace_data_dependency_supports_repeated_weights():
    data = _pod_dependency_data()
    weight = pt.tensor([1.0, 2.0, 3.0, 4.0])

    result = pod_subspace_data_dependency(
        data,
        ranks=(1, 2),
        sequence_fractions=(0.5, 1.0),
        snapshot_strides=(1,),
        weight=weight,
        mode="svd",
    )

    assert bool(pt.isfinite(result.similarity).all())
    pt.testing.assert_close(result.similarity[-1, 0], pt.ones(2))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence_fractions": (0.0,)}, "sequence_fractions"),
        ({"snapshot_strides": (1.5,)}, "snapshot_strides"),
        (
            {
                "ranks": (1, 5),
                "sequence_fractions": (0.2,),
                "snapshot_strides": (2,),
            },
            "available rank",
        ),
    ],
)
def test_pod_subspace_data_dependency_validates_sweep(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pod_subspace_data_dependency(_pod_dependency_data(), mode="svd", **kwargs)


class TestSVD:
    def _assert_compact_storage(self, tensor):
        expected = tensor.element_size() * tensor.nelement()
        assert tensor.untyped_storage().nbytes() == expected

    def test_init_cpu(self):
        M, N = 100, 15
        dm = pt.rand((M, N), dtype=pt.float32)
        svd = SVD(dm, rank=N)
        assert svd.U.shape == (M, N)
        assert svd.V.shape == (N, N)
        assert svd.s.shape == (N,)
        assert svd.s_rel.shape == (N,)
        assert svd.s_cum.shape == (N,)
        assert svd.mode == "evd"
        svd = SVD(dm, mode="svd")
        assert svd.mode == "svd"
        assert svd.opt_rank == svd.rank
        svd = SVD(pt.rand((N, N)))
        assert svd.mode == "svd"
        svd = SVD(dm, rank=M)
        assert svd.rank == N
        dm = pt.rand((N, M), dtype=pt.float32)
        svd = SVD(dm, rank=N)
        assert svd.mode == "evd"
        with pytest.raises(ValueError):
            _ = SVD(dm.unsqueeze(-1))
        with pytest.raises(ValueError):
            _ = SVD(dm, mode="abc")

    def test_source_backed_tsqr_reconstructs_out_of_core(self):
        pt.manual_seed(7)
        data = pt.rand((41, 6), dtype=pt.float64)
        source = _MatrixSource(data)

        svd = SVD(
            source,
            rank=6,
            spatial_batch_size=7,
            snapshot_batch_size=2,
        )

        assert svd.mode == "tsqr"
        assert svd.U.global_shape == (41, 6)
        pt.testing.assert_close(svd.reconstruct().materialize_local(), data)
        assert all(call[0].stop - call[0].start <= 7 for call in source.calls)
        assert all(call[1].stop - call[1].start <= 2 for call in source.calls)

    def test_source_backed_centered_weighted_tsqr(self):
        pt.manual_seed(8)
        data = pt.rand((37, 5), dtype=pt.float64)
        weight = pt.linspace(1.0, 2.0, data.shape[0], dtype=pt.float64)
        source = _MatrixSource(data, weight)

        svd = SVD(source, rank=4, subtract_mean=True, spatial_batch_size=8)
        modes = svd.U.materialize_local()
        mean = svd.mean.materialize_local()

        pt.testing.assert_close(mean, data.mean(dim=1))
        pt.testing.assert_close(
            modes.conj().T @ (modes * weight.unsqueeze(-1)), pt.eye(4, dtype=data.dtype)
        )
        pt.testing.assert_close(
            svd.reconstruct().materialize_local(),
            data - data.mean(dim=1, keepdim=True),
        )

    def test_source_backed_incremental_update_matches_direct_svd(self):
        pt.manual_seed(9)
        first = pt.rand((43, 4), dtype=pt.float64)
        second = pt.rand((43, 3), dtype=pt.float64)
        svd = SVD(_MatrixSource(first), rank=4, spatial_batch_size=9)

        svd.update(_MatrixSource(second))

        combined = pt.cat((first, second), dim=1)
        direct = SVD(combined, rank=4, mode="svd")
        pt.testing.assert_close(svd.s, direct.s)
        pt.testing.assert_close(
            subspace_similarity(svd.U.materialize_local(), direct.U, ranks=4),
            pt.ones(1, dtype=first.dtype),
        )

    def test_centered_update_recomputes_with_combined_mean(self):
        pt.manual_seed(10)
        first = pt.rand((31, 4), dtype=pt.float64)
        second = 2.0 + pt.rand((31, 2), dtype=pt.float64)
        svd = SVD(
            _MatrixSource(first), rank=5, subtract_mean=True, spatial_batch_size=6
        )

        svd.update(_MatrixSource(second))

        combined = pt.cat((first, second), dim=1)
        centered = combined - combined.mean(dim=1, keepdim=True)
        direct = SVD(centered, rank=4, mode="svd")
        pt.testing.assert_close(
            svd.reconstruct().materialize_local(),
            direct.reconstruct(),
        )

    def test_tensor_checkpoint_round_trip(self, tmp_path):
        pt.manual_seed(14)
        data = pt.rand((18, 5), dtype=pt.float64)
        weight = pt.linspace(1.0, 2.0, data.shape[0], dtype=data.dtype)
        svd = SVD(
            data,
            rank=4,
            mode="svd",
            weight=weight,
            subtract_mean=True,
        )
        path = tmp_path / "tensor-svd.pt"

        svd.save(path)
        restored = SVD.load(path)

        assert restored.rank == svd.rank
        assert restored.opt_rank == svd.opt_rank
        assert restored.mode == svd.mode
        pt.testing.assert_close(restored.U, svd.U)
        pt.testing.assert_close(restored.s_full, svd.s_full)
        pt.testing.assert_close(restored.V, svd.V)
        pt.testing.assert_close(restored.mean, svd.mean)
        pt.testing.assert_close(restored.reconstruct(), svd.reconstruct())

    def test_source_checkpoint_reconnects_without_reading_data(self, tmp_path):
        pt.manual_seed(15)
        data = pt.rand((29, 6), dtype=pt.float64)
        source = _MatrixSource(data)
        svd = SVD(
            source,
            rank=5,
            subtract_mean=True,
            spatial_batch_size=7,
            snapshot_batch_size=2,
        )
        path = tmp_path / "source-svd.pt"
        svd.save(path)
        restored_source = _MatrixSource(data)

        restored = SVD.load(path, source=restored_source)

        assert restored_source.calls == []
        pt.testing.assert_close(restored.s_full, svd.s_full)
        pt.testing.assert_close(restored.V, svd.V)
        pt.testing.assert_close(
            restored.reconstruct().materialize_local(),
            data - data.mean(dim=1, keepdim=True),
        )

    def test_source_checkpoint_rejects_incompatible_source(self, tmp_path):
        svd = SVD(_MatrixSource(pt.rand((12, 4))), rank=3)
        path = tmp_path / "source-svd.pt"
        svd.save(path)

        with pytest.raises(ValueError, match="incompatible"):
            SVD.load(path, source=_MatrixSource(pt.rand((13, 4))))

    def test_truncated_factors_use_compact_storage(self):
        M, N, rank = 100, 15, 3
        for mode in ("svd", "evd"):
            svd = SVD(pt.rand((M, N), dtype=pt.float32), rank=rank, mode=mode)
            self._assert_compact_storage(svd.U)
            self._assert_compact_storage(svd.s)
            self._assert_compact_storage(svd.V)
            assert svd.U.is_contiguous()
            assert svd.s.is_contiguous()
            assert svd.V.is_contiguous()

    @pytest.mark.skipif(not pt.cuda.is_available(), reason="CUDA not available")
    def test_init_gpu(self):
        M, N = 100, 15
        dm = pt.rand((M, N), dtype=pt.float32, device="cuda")
        svd = SVD(dm, rank=N)
        assert svd.U.shape == (M, N)
        assert "cuda" in str(svd.U.device)
        assert svd.V.shape == (N, N)
        assert svd.s.shape == (N,)
        assert svd.s_rel.shape == (N,)
        assert svd.s_cum.shape == (N,)
        assert svd.mode == "evd"
        svd = SVD(dm, mode="svd")
        assert svd.mode == "svd"
        assert svd.opt_rank == svd.rank
        svd = SVD(pt.rand((N, N)))
        assert svd.mode == "svd"
        svd = SVD(dm, rank=M)
        assert svd.rank == N

    def test_reconstruct(self):
        M, N = 100, 15
        dm = pt.rand((M, N), dtype=pt.float32)
        svd = SVD(dm, N)
        pt.testing.assert_close(dm, svd.reconstruct())
        err_r1 = pt.linalg.norm(dm - svd.reconstruct(rank=1)).item()
        err_r2 = pt.linalg.norm(dm - svd.reconstruct(rank=2)).item()
        assert err_r2 <= err_r1

    @pytest.mark.parametrize("complex_data", [False, True])
    def test_weighted_svd(self, complex_data):
        pt.manual_seed(0)
        data = pt.rand((6, 4), dtype=pt.float64)
        if complex_data:
            data = data + 1j * pt.rand_like(data)
        weight = pt.tensor([1.0, 2.0, 4.0], dtype=pt.float64)
        expanded_weight = weight.repeat(2)

        svd = SVD(data, rank=4, mode="svd", weight=weight)

        assert svd.weight is not None
        pt.testing.assert_close(svd.weight, expanded_weight)
        weighted_gram = svd.U.conj().T @ (expanded_weight.unsqueeze(-1) * svd.U)
        pt.testing.assert_close(weighted_gram, pt.eye(4, dtype=data.dtype))
        pt.testing.assert_close(
            svd.U_weighted.conj().T @ svd.U_weighted,
            pt.eye(4, dtype=data.dtype),
        )
        pt.testing.assert_close(svd.reconstruct(), data)
        pt.testing.assert_close(
            svd.s,
            pt.linalg.svdvals(expanded_weight.sqrt().unsqueeze(-1) * data),
        )

    def test_unit_weight_matches_unweighted_svd(self):
        data = pt.rand((6, 4), dtype=pt.float64)
        unweighted = SVD(data, rank=4, mode="svd")
        weighted = SVD(data, rank=4, mode="svd", weight=pt.ones(6))

        pt.testing.assert_close(weighted.s, unweighted.s)
        pt.testing.assert_close(
            subspace_similarity(weighted.U, unweighted.U),
            pt.ones(4, dtype=data.dtype),
        )
        assert unweighted.weight is None
        assert unweighted.U_weighted is unweighted.U

    @pytest.mark.parametrize(
        ("weight", "message"),
        [
            (pt.tensor([]), "non-empty"),
            (pt.ones(2), "must divide"),
            (pt.tensor([1.0, 0.0, 1.0]), "positive"),
            (pt.tensor([1.0, float("inf"), 1.0]), "finite"),
            (pt.ones(3, dtype=pt.complex64), "real numeric"),
        ],
    )
    def test_weighted_svd_validates_weight(self, weight, message):
        with pytest.raises(ValueError, match=message):
            SVD(pt.rand((5, 3)), weight=weight)
