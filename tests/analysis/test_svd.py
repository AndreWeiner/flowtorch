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
