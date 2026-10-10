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


def test_available_memory_respects_cgroup_limit(monkeypatch):
    from pathlib import Path
    from flowtorch.analysis.svd import _available_memory_bytes

    contents = {
        "/proc/meminfo": "MemAvailable: 1024 kB\n",
        "/proc/self/cgroup": "0::/worker\n",
        "/sys/fs/cgroup/worker/memory.max": "1000",
        "/sys/fs/cgroup/worker/memory.current": "200",
        "/sys/fs/cgroup/memory.max": "600",
        "/sys/fs/cgroup/memory.current": "100",
    }
    monkeypatch.setattr(Path, "read_text", lambda path: contents[str(path)])
    assert _available_memory_bytes(pt.device("cpu")) == 500


def test_available_cuda_memory_uses_device(monkeypatch):
    from flowtorch.analysis.svd import _available_memory_bytes

    devices = []

    def memory_info(device):
        devices.append(device)
        return (1234, 5678)

    monkeypatch.setattr(pt.cuda, "mem_get_info", memory_info)
    device = pt.device("cuda:2")
    assert _available_memory_bytes(device) == 1234
    assert devices == [device]


@pytest.mark.parametrize(
    "hosts,available,device_type,explicit,expected_budget",
    [
        # One workstation: all four ranks share RAM.
        (["a"] * 4, [800_000] * 4, "cpu", None, 120_000),
        # Two nodes: only the two ranks on each host share that host's RAM.
        (["a", "a", "b", "b"], [800_000] * 4, "cpu", None, 240_000),
        # The more constrained node bounds the common size.
        (
            ["a", "a", "b", "b"],
            [800_000, 800_000, 400_000, 400_000],
            "cpu",
            None,
            120_000,
        ),
        # Explicit budgets are already per process, without the safety factor.
        (["a"] * 4, [None] * 4, "cpu", [240_000, 240_000, 120_000, 240_000], 120_000),
        # One GPU per rank: device memory is not shared by hostname.
        (["a"] * 4, [800_000] * 4, "cuda", None, 480_000),
    ],
)
def test_initial_capacity_accounts_for_host_memory_sharing(
    hosts, available, device_type, explicit, expected_budget
):
    from flowtorch.analysis.svd import _initial_memory_capacity

    reports = [
        dict(
            hostname=host,
            available=memory,
            device_type=device_type,
            budget=None if explicit is None else explicit[index],
            scalar_bytes=8,
            rows_per_value=1,
        )
        for index, (host, memory) in enumerate(zip(hosts, available))
    ]
    # Four snapshots; factors for four ranks: 1280 bytes;
    # workspace per spatial value: 192 bytes.
    expected = (expected_budget - 5632) // 448
    assert _initial_memory_capacity(reports, 4) == expected


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
        )

        assert svd.mode == "tsqr"
        assert svd.U.global_shape == (41, 6)
        pt.testing.assert_close(svd.reconstruct().materialize_local(), data)
        assert all(call[0].stop - call[0].start <= 7 for call in source.calls)
        assert all(call[1] == slice(0, 6) for call in source.calls)

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

    def test_encode_and_decode_use_centering_and_weighted_inner_product(self):
        pt.manual_seed(16)
        data = pt.rand((7, 5), dtype=pt.float64)
        weight = pt.linspace(1.0, 2.0, data.shape[0], dtype=data.dtype)
        svd = SVD(
            data,
            rank=5,
            mode="svd",
            weight=weight,
            subtract_mean=True,
        )

        pt.testing.assert_close(svd.decode(svd.encode(data)), data)
        coefficients = pt.rand((5, 2, 3), dtype=data.dtype)
        assert svd.decode(coefficients).shape == (7, 2, 3)

    def test_source_encode_and_decode_arbitrary_snapshots(self):
        pt.manual_seed(17)
        data = pt.rand((11, 5), dtype=pt.float64)
        source = _MatrixSource(data)
        svd = SVD(source, rank=4, subtract_mean=True, spatial_batch_size=3)

        coefficients = svd.encode(_MatrixSource(data))
        decoded = svd.decode(coefficients).materialize_local()

        pt.testing.assert_close(decoded, data)

    @pytest.mark.parametrize("complex_data", [False, True])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_source_backed_incremental_update_matches_direct_svd(
        self, complex_data, weighted
    ):
        pt.manual_seed(9)
        first = pt.rand((43, 4), dtype=pt.float64)
        second = pt.rand((43, 3), dtype=pt.float64)
        if complex_data:
            first = first + 1j * pt.rand_like(first)
            second = second + 1j * pt.rand_like(second)
        weight = pt.linspace(1.0, 2.0, 43, dtype=pt.float64) if weighted else None
        svd = SVD(_MatrixSource(first, weight), rank=4, spatial_batch_size=9)

        svd.update(_MatrixSource(second, weight))

        combined = pt.cat((first, second), dim=1)
        direct = SVD(combined, rank=4, mode="svd", weight=weight)
        pt.testing.assert_close(svd.s, direct.s)
        pt.testing.assert_close(
            subspace_similarity(
                svd.U.materialize_local(), direct.U, ranks=4, weight=weight
            ),
            pt.ones(1, dtype=first.real.dtype),
        )
        pt.testing.assert_close(
            svd.reconstruct().materialize_local(), direct.reconstruct()
        )
        self._assert_compact_storage(svd.V)
        assert svd.V.is_contiguous()

    def test_centered_update_tracks_combined_mean(self):
        pt.manual_seed(10)
        first = pt.rand((31, 4), dtype=pt.float64)
        second = 2.0 + pt.rand((31, 2), dtype=pt.float64)
        svd = SVD(
            _MatrixSource(first), rank=5, subtract_mean=True, spatial_batch_size=6
        )

        svd.update(_MatrixSource(second))

        combined = pt.cat((first, second), dim=1)
        centered = combined - combined.mean(dim=1, keepdim=True)
        direct = SVD(centered, rank=5, mode="svd")
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

    @pytest.mark.parametrize("shape", [(80, 12), (12, 80)])
    @pytest.mark.parametrize("complex_data", [False, True])
    @pytest.mark.parametrize("rank", [None, 3, 12])
    def test_evd_truncates_before_forming_large_factor(self, shape, complex_data, rank):
        pt.manual_seed(24)
        data = pt.randn(shape, dtype=pt.float64)
        if complex_data:
            data = data + 1j * pt.randn_like(data)
        weight = pt.linspace(1.0, 2.0, shape[0], dtype=pt.float64)
        with pt.profiler.profile(record_shapes=True) as profile:
            svd = SVD(data, rank=rank, mode="evd", weight=weight, subtract_mean=True)
        direct = SVD(data, rank=svd.rank, mode="svd", weight=weight, subtract_mean=True)
        pt.testing.assert_close(svd.s_full, direct.s_full, atol=1e-6, rtol=1e-6)
        pt.testing.assert_close(svd.reconstruct(), direct.reconstruct())
        assert svd.opt_rank == direct.opt_rank
        self._assert_compact_storage(svd.U)
        self._assert_compact_storage(svd.V)
        # The second matrix multiplication forms only the retained columns.
        multiplications = [
            event for event in profile.events() if event.name == "aten::mm"
        ]
        assert len(multiplications) == 2
        assert multiplications[1].input_shapes[1][-1] == svd.rank

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

    @pytest.mark.parametrize("shape", [(25, 6), (6, 25)])
    @pytest.mark.parametrize("complex_data", [False, True])
    def test_reconstruct_scaling_preserves_results_and_gradients(
        self, shape, complex_data
    ):
        pt.manual_seed(25)
        data = pt.randn(shape, dtype=pt.float64)
        if complex_data:
            data = data + 1j * pt.randn_like(data)
        svd = SVD(data, rank=6, mode="svd")
        svd._U = svd.U.detach().requires_grad_()
        svd._V = svd.V.detach().requires_grad_()
        svd._s = svd.s.detach().requires_grad_()
        for rank in (2, 6):
            expected = (svd.U[:, :rank] * svd.s[:rank]) @ svd.V[:, :rank].conj().T
            actual = svd.reconstruct(rank)
            pt.testing.assert_close(actual, expected)
            factors = (svd.U, svd.s, svd.V)
            expected_grad = pt.autograd.grad(expected.abs().square().sum(), factors)
            actual_grad = pt.autograd.grad(actual.abs().square().sum(), factors)
            for actual_value, expected_value in zip(actual_grad, expected_grad):
                pt.testing.assert_close(actual_value, expected_value)

    @pytest.mark.parametrize("batch_size", [3, 20])
    @pytest.mark.parametrize("complex_data", [False, True])
    def test_source_reconstruct_forms_only_requested_modes(
        self, batch_size, complex_data
    ):
        pt.manual_seed(26)
        data = pt.randn((20, 6), dtype=pt.float64)
        if complex_data:
            data = data + 1j * pt.randn_like(data)
        svd = SVD(_MatrixSource(data), rank=6, spatial_batch_size=batch_size)
        spatial_slice = slice(0, batch_size)
        chunk = svd._left_chunk(spatial_slice, rank=2)
        expected = (data[spatial_slice] @ svd.V[:, :2]) / svd.s[:2]
        pt.testing.assert_close(chunk, expected)
        reconstructed = svd.reconstruct(rank=2).materialize_local()
        pt.testing.assert_close(
            reconstructed, (data @ svd.V[:, :2]) @ svd.V[:, :2].conj().T
        )

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

    @pytest.mark.parametrize("mode", ["svd", "evd"])
    @pytest.mark.parametrize("complex_data", [False, True])
    def test_centered_weighted_initialization_preserves_input(self, mode, complex_data):
        pt.manual_seed(27)
        data = pt.randn((24, 5), dtype=pt.float64)
        if complex_data:
            data = data + 1j * pt.randn_like(data)
        original = data.clone()
        weight = pt.linspace(1.0, 2.0, data.shape[0], dtype=pt.float64)
        original_weight = weight.clone()
        svd = SVD(data, rank=5, mode=mode, weight=weight, subtract_mean=True)
        pt.testing.assert_close(data, original, rtol=0, atol=0)
        pt.testing.assert_close(weight, original_weight, rtol=0, atol=0)
        pt.testing.assert_close(svd.mean, original.mean(dim=1))
        centered = original - original.mean(dim=1, keepdim=True)
        pt.testing.assert_close(svd.reconstruct(), centered)
        pt.testing.assert_close(
            svd.s_full, pt.linalg.svdvals(centered * weight.sqrt().unsqueeze(-1))
        )

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


@pytest.mark.parametrize("storage", ["disk", "memory"])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
def test_stored_basis_updates_without_old_snapshots(
    tmp_path, storage, centered, complex_data
):
    pt.manual_seed(42)
    data = pt.randn((37, 12), dtype=pt.float64)
    if complex_data:
        data = data + 1j * pt.randn_like(data)
    data[:, 4:] += 4.0
    weight = pt.linspace(0.5, 2.0, 37, dtype=data.real.dtype)
    first = _MatrixSource(data[:, :4], weight)
    svd = SVD(
        first,
        rank=100,
        subtract_mean=centered,
        spatial_batch_size=7,
        basis_storage=storage,
        basis_directory=tmp_path,
    )
    assert svd.maximum_rank == 100
    first.calls.clear()
    first.data = None  # Old snapshots are deliberately inaccessible.
    for stop in (8, 12):
        batch = _MatrixSource(data[:, stop - 4 : stop], weight)
        svd.update(batch)
        batch.data = None
        direct = SVD(
            data[:, :stop],
            rank=stop - int(centered),
            mode="svd",
            subtract_mean=centered,
            weight=weight,
        )
        assert svd.rank == stop - int(centered)
        pt.testing.assert_close(svd.s, direct.s)
        pt.testing.assert_close(
            svd.reconstruct().materialize_local(), direct.reconstruct()
        )
        modes = svd.U.materialize_local()
        pt.testing.assert_close(
            modes.conj().T @ (weight[:, None] * modes),
            pt.eye(svd.rank, dtype=data.dtype),
        )
        if centered:
            pt.testing.assert_close(
                svd.mean.materialize_local(), data[:, :stop].mean(dim=1)
            )
            pt.testing.assert_close(
                svd.decode(
                    svd.encode(_MatrixSource(data[:, :stop], weight))
                ).materialize_local(),
                data[:, :stop],
            )
    assert first.calls == []
    checkpoint = tmp_path / "checkpoint"
    svd.save(checkpoint)
    expected = svd.reconstruct().materialize_local()
    svd.close()
    restored = SVD.load(
        checkpoint,
        basis_storage=storage,
        basis_directory=tmp_path,
        spatial_batch_size=5,
    )
    assert restored.maximum_rank == 100
    pt.testing.assert_close(restored.reconstruct().materialize_local(), expected)
    restored.update(_MatrixSource(data[:, :2], weight))
    combined = pt.cat((data, data[:, :2]), dim=1)
    direct = SVD(
        combined,
        rank=14 - int(centered),
        mode="svd",
        subtract_mean=centered,
        weight=weight,
    )
    pt.testing.assert_close(
        restored.reconstruct().materialize_local(), direct.reconstruct()
    )
    restored.close()
    assert list(tmp_path.iterdir()) == [checkpoint]


@pytest.mark.parametrize("storage", ["disk", "memory"])
def test_maximum_rank_grows_then_caps(storage):
    pt.manual_seed(43)
    data = pt.randn((40, 15), dtype=pt.float64)
    svd = SVD(_MatrixSource(data[:, :3]), rank=7, basis_storage=storage)
    assert svd.rank == 3
    for stop, expected in ((6, 6), (9, 7), (12, 7), (15, 7)):
        svd.update(_MatrixSource(data[:, stop - 3 : stop]))
        assert svd.rank == expected
        assert svd.maximum_rank == 7
        assert svd.V.shape == (stop, expected)
        modes = svd.U.materialize_local()
        pt.testing.assert_close(modes.T @ modes, pt.eye(expected, dtype=data.dtype))


def test_disk_basis_checkpoint_does_not_materialize_complete_basis(
    tmp_path, monkeypatch
):
    from flowtorch.analysis.svd import _BasisStore

    data = pt.randn((41, 8), dtype=pt.float64)
    svd = SVD(
        _MatrixSource(data), rank=4, spatial_batch_size=6, basis_directory=tmp_path
    )
    original_read = _BasisStore.read

    def bounded_read(self, spatial_slice, *args):
        assert spatial_slice.stop - spatial_slice.start <= 6
        return original_read(self, spatial_slice, *args)

    monkeypatch.setattr(_BasisStore, "read", bounded_read)
    svd.update(_MatrixSource(data[:, :2]))
    checkpoint = tmp_path / "checkpoint"
    svd.save(checkpoint)
    with pytest.raises(FileExistsError):
        svd.save(checkpoint)
    restored = SVD.load(checkpoint)
    pt.testing.assert_close(restored.U.materialize_local(), svd.U.materialize_local())


def test_update_inherits_explicit_weight_and_preserves_state_on_failure(tmp_path):
    data = pt.randn((17, 4), dtype=pt.float64)
    weight = pt.linspace(1.0, 2.0, 17, dtype=data.dtype)
    svd = SVD(
        _MatrixSource(data[:, :2]),
        rank=4,
        weight=weight,
        spatial_batch_size=5,
        basis_directory=tmp_path,
    )
    svd.update(_MatrixSource(data[:, 2:]))
    direct = SVD(data, rank=4, mode="svd", weight=weight)
    pt.testing.assert_close(svd.reconstruct().materialize_local(), direct.reconstruct())
    before = svd.reconstruct().materialize_local()
    with pytest.raises(RuntimeError, match="weights are incompatible"):
        svd.update(_MatrixSource(data, weight * 2))
    pt.testing.assert_close(svd.reconstruct().materialize_local(), before)


@pytest.mark.parametrize("storage", ["disk", "memory"])
def test_basis_chunks_preserve_component_major_layout(tmp_path, storage):
    class ComponentSource(StateVectorSource):
        def __init__(self, data):
            self.data = data
            self._layout = StateVectorLayout(
                (FieldSpec("u", (2,)), FieldSpec("p")), (11, 2)
            )

        @property
        def layout(self):
            return self._layout

        @property
        def n_snapshots(self):
            return self.data.shape[-1]

        def read(self, spatial_slice, snapshot_slice):
            return self.data[:, spatial_slice, :, snapshot_slice].reshape(
                -1, self.data[:, spatial_slice, :, snapshot_slice].shape[-1]
            )

    pt.manual_seed(45)
    data = pt.randn((3, 11, 2, 7), dtype=pt.float64)
    svd = SVD(
        ComponentSource(data[..., :3]),
        rank=7,
        subtract_mean=True,
        spatial_batch_size=4,
        basis_storage=storage,
    )
    svd.update(ComponentSource(data[..., 3:]))
    direct = SVD(data.reshape(66, 7), rank=6, mode="svd", subtract_mean=True)
    pt.testing.assert_close(svd.reconstruct().materialize_local(), direct.reconstruct())
    path = tmp_path / "components"
    svd.save(path)
    restored = SVD.load(path, spatial_batch_size=3, basis_storage=storage)
    pt.testing.assert_close(restored.U.materialize_local(), svd.U.materialize_local())
    pt.testing.assert_close(
        restored.mean.materialize_local(), data.mean(dim=-1).reshape(-1)
    )


def test_disk_restore_copies_arrays_without_materializing(tmp_path, monkeypatch):
    from flowtorch.analysis import svd as module

    svd = SVD(_MatrixSource(pt.randn((25, 5), dtype=pt.float64)), rank=3)
    checkpoint = tmp_path / "checkpoint"
    svd.save(checkpoint)
    with monkeypatch.context() as patch:
        patch.setattr(
            module.np,
            "load",
            lambda *args, **kwargs: pytest.fail("disk restore materialized an array"),
        )
        restored = SVD.load(checkpoint, basis_storage="disk")
    pt.testing.assert_close(restored.U.materialize_local(), svd.U.materialize_local())


@pytest.mark.parametrize("centered", [False, True])
def test_single_snapshot_updates_include_mean_correction(centered):
    pt.manual_seed(46)
    data = pt.randn((21, 5), dtype=pt.float64)
    svd = SVD(_MatrixSource(data[:, :1]), rank=10, subtract_mean=centered)
    for column in range(1, 5):
        svd.update(_MatrixSource(data[:, column : column + 1]))
    direct = SVD(data, rank=5 - int(centered), mode="svd", subtract_mean=centered)
    pt.testing.assert_close(svd.reconstruct().materialize_local(), direct.reconstruct())
    if centered:
        pt.testing.assert_close(svd.mean.materialize_local(), data.mean(dim=1))


def test_automatic_rank_remains_automatic_on_update():
    pt.manual_seed(47)
    data = pt.randn((25, 8), dtype=pt.float64)
    svd = SVD(_MatrixSource(data[:, :4]))
    assert svd.maximum_rank is None
    svd.update(_MatrixSource(data[:, 4:]))
    assert svd.rank == max(1, svd.opt_rank)
    assert svd.V.shape == (8, svd.rank)


@pytest.mark.parametrize("storage", ["disk", "memory"])
def test_update_reduces_size_only_for_memory_capacity(monkeypatch, storage):
    data = pt.randn((31, 8), dtype=pt.float64)
    svd = SVD(
        _MatrixSource(data[:, :4]), rank=8, spatial_batch_size=7, basis_storage=storage
    )
    # Four new columns, rank grows from four to eight. The budget fits three rows.
    fixed = 7 * 8**2 * 8 + 2 * 8 * 8 * 8
    if storage == "memory":
        fixed += 31 * 8 * (8 + 4 + 1)
    per_value = 8 * (6 * 4 + 4 * (4 + 8)) + 2 * 8 * (4 + 4 + 2)
    monkeypatch.setattr(
        "flowtorch.analysis.svd._available_memory_bytes",
        lambda device: (fixed + 3 * per_value + 1) / 0.6,
    )
    svd.update(_MatrixSource(data[:, 4:]))
    assert svd._spatial_batch_size == 3
    pt.testing.assert_close(svd.reconstruct().materialize_local(), data)
    monkeypatch.setattr(
        "flowtorch.analysis.svd._available_memory_bytes", lambda device: 10**9
    )
    svd.update(_MatrixSource(data[:, :1]))
    assert svd._spatial_batch_size == 3  # Capacity checks do not tune or grow it.


def test_update_rejects_insufficient_memory_before_reading(monkeypatch):
    data = pt.randn((21, 5), dtype=pt.float64)
    svd = SVD(_MatrixSource(data[:, :3]), rank=5, spatial_batch_size=7)
    source = _MatrixSource(data[:, 3:])
    before = svd.reconstruct().materialize_local()
    monkeypatch.setattr(
        "flowtorch.analysis.svd._available_memory_bytes", lambda device: 1
    )
    with pytest.raises(ValueError, match="update memory budget is too small"):
        svd.update(source)
    assert source.calls == []
    assert svd._spatial_batch_size == 7
    pt.testing.assert_close(svd.reconstruct().materialize_local(), before)


@pytest.mark.parametrize("storage", ["disk", "memory"])
@pytest.mark.parametrize("centered", [False, True])
def test_prefetched_update_matches_sequential_and_checkpoint(
    tmp_path, storage, centered
):
    pt.manual_seed(53)
    data = pt.randn((43, 9), dtype=pt.float64)
    weight = pt.linspace(0.5, 2.0, 43, dtype=data.dtype)
    results = []
    for enabled in (False, True):
        svd = SVD(
            _MatrixSource(data[:, :3], weight),
            rank=9,
            spatial_batch_size=7,
            subtract_mean=centered,
            basis_storage=storage,
            prefetch_chunks=enabled,
        )
        for start in (3, 6):
            svd.update(_MatrixSource(data[:, start : start + 3], weight))
        results.append(svd)
    pt.testing.assert_close(results[0].s, results[1].s)
    pt.testing.assert_close(
        results[0].reconstruct().materialize_local(),
        results[1].reconstruct().materialize_local(),
    )
    path = tmp_path / "checkpoint"
    results[1].save(path)
    restored = SVD.load(path, prefetch_chunks=False)
    assert restored._prefetch_chunks is False
    pt.testing.assert_close(
        restored.U.materialize_local(), results[1].U.materialize_local()
    )


@pytest.mark.parametrize("dtype", [pt.float32, pt.float64])
def test_default_spatial_target_has_no_trial_reads(monkeypatch, dtype):
    data = pt.randn((41, 5), dtype=dtype)
    source = _MatrixSource(data)
    monkeypatch.setattr(
        "flowtorch.analysis.svd.SVD_DEFAULT_INPUT_BYTES", 8 * 5 * data.element_size()
    )
    svd = SVD(source, rank=5, subtract_mean=True)
    assert svd._spatial_batch_size == 8
    assert svd._prefetch_chunks is False
    assert source.calls[0] == (slice(0, 1), slice(0, 1))
    assert len(source.calls) == 1 + 2 * 6
    direct = SVD(data, rank=4, subtract_mean=True, mode="svd")
    pt.testing.assert_close(svd.s, direct.s)
    pt.testing.assert_close(svd.reconstruct().materialize_local(), direct.reconstruct())
    svd.close()


def test_default_memory_bound_reserves_maximum_rank(monkeypatch):
    data = pt.randn((200, 4), dtype=pt.float64)
    small = SVD(_MatrixSource(data), rank=4, memory_budget_bytes=1_000_000)
    large = SVD(_MatrixSource(data), rank=100, memory_budget_bytes=1_000_000)
    assert large._spatial_batch_size < small._spatial_batch_size
    queued = SVD(
        _MatrixSource(data),
        rank=100,
        memory_budget_bytes=1_000_000,
        prefetch_chunks=True,
    )
    assert queued._spatial_batch_size < large._spatial_batch_size
    for svd in (small, large, queued):
        pt.testing.assert_close(svd.reconstruct().materialize_local(), data)
        svd.close()


def test_explicit_spatial_size_overrides_input_target(monkeypatch):
    monkeypatch.setattr("flowtorch.analysis.svd.SVD_DEFAULT_INPUT_BYTES", 1)
    source = _MatrixSource(pt.randn((17, 4)))
    svd = SVD(source, rank=4, spatial_batch_size=7)
    assert svd._spatial_batch_size == 7
    assert source.calls[0] == (slice(0, 7), slice(0, 4))
    svd.close()


def test_default_rejects_insufficient_memory():
    with pytest.raises(ValueError, match="too small"):
        SVD(_MatrixSource(pt.randn((17, 4))), memory_budget_bytes=1)


def test_removed_search_keyword_is_not_accepted():
    with pytest.raises(TypeError, match="autotune_chunks"):
        SVD(_MatrixSource(pt.randn((17, 4))), autotune_chunks=True)


def test_update_reuses_default_without_reselecting(monkeypatch):
    data = pt.randn((31, 8), dtype=pt.float64)
    monkeypatch.setattr("flowtorch.analysis.svd.SVD_DEFAULT_INPUT_BYTES", 7 * 4 * 8)
    svd = SVD(_MatrixSource(data[:, :4]), rank=8)
    assert svd._spatial_batch_size == 7
    monkeypatch.setattr(
        SVD, "_select_spatial_batch", lambda *args: pytest.fail("update reselected")
    )
    svd.update(_MatrixSource(data[:, 4:]))
    assert svd._spatial_batch_size == 7
    pt.testing.assert_close(svd.reconstruct().materialize_local(), data)
    svd.close()
