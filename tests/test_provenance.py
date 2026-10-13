"""Contracts for shared provenance and existing computation entry points."""

import json

import pytest
import torch as pt

from flowtorch.analysis import DFT, PDFT, SVD
from flowtorch.provenance import (
    ProvenanceMixin,
    ProvenanceSession,
    describe,
    tracked,
)
from flowtorch.rom import DMD


class Example(ProvenanceMixin):
    @tracked
    def __init__(self, value=2):
        self.value = value

    def _effective_settings(self):
        return {"value": self.value}

    @tracked
    def change(self, value):
        if value < 0:
            raise ValueError("invalid")
        self.value = value


class Derived(Example):
    @tracked
    def __init__(self, value=3):
        super().__init__(value)


def test_defaults_history_and_failed_operations(tmp_path):
    obj = Derived()
    assert obj.settings == {"value": 3}
    assert len(obj.provenance["records"]) == 1
    obj.change(4)
    with pytest.raises(ValueError):
        obj.change(-1)
    records = obj.provenance["records"]
    assert records[0]["effective"]["value"] == 3
    assert records[1]["effective"]["value"] == 4
    assert records[2]["status"] == "failed"
    records[0]["requested"]["value"] = 100
    assert obj.settings["value"] == 3
    obj.dump_provenance(tmp_path / "metadata.json")
    assert json.loads((tmp_path / "metadata.json").read_text())["current"] == {
        "value": 4
    }


def test_svd_capture_rng_checkpoint_and_old_checkpoint(tmp_path):
    data = pt.arange(24, dtype=pt.float64).reshape(6, 4)
    rng = pt.get_rng_state().clone()
    svd = SVD(data, rank=2, mode="svd")
    assert pt.equal(rng, pt.get_rng_state())
    assert svd.settings["subtract_mean"] is False
    assert svd.provenance["current"]["mode"] == "svd"
    path = tmp_path / "svd.pt"
    svd.save(path)
    restored = SVD.load(path)
    assert restored.provenance["records"] == svd.provenance["records"]
    assert pt.allclose(restored.reconstruct(), svd.reconstruct())
    checkpoint = pt.load(path, weights_only=True)
    checkpoint.pop("provenance")
    pt.save(checkpoint, path)
    assert SVD.load(path).versions == {}
    svd.dump_versions(tmp_path / "versions.json")
    assert json.loads((tmp_path / "versions.json").read_text())["torch"] == str(
        pt.__version__
    )


def test_factory_and_dependency_snapshot():
    svd = SVD(pt.arange(24, dtype=pt.float64).reshape(6, 4), rank=2)
    spectrum = PDFT.from_svd(svd, dt=0.1)
    record = spectrum.provenance["records"][-1]
    assert record["operation"].endswith("from_svd")
    assert record["requested"]["svd"]["provenance"]["current"]["rank"] == 2
    svd.rank = 1
    assert record == spectrum.provenance["records"][-1]


def test_rom_fit_stages():
    from flowtorch.rom import Trajectory

    model = DMD(rank=2)
    model.fit(Trajectory(pt.rand(6, 8), pt.arange(8, dtype=pt.float32)))
    assert len(model.provenance["records"]) == 2
    assert model.provenance["current"]["svd"]["provenance"]["records"]


def test_no_lazy_properties_or_tensor_contents():
    class Lazy:
        @property
        def values(self):
            raise AssertionError("must not read")

    assert describe(Lazy())["configuration_recorded"] is False
    value = pt.ones(3, requires_grad=True)
    assert describe(value) == {
        "kind": "tensor",
        "shape": [3],
        "dtype": "torch.float32",
        "device": "cpu",
        "content_recorded": False,
    }


def test_visualization_session_keeps_returns_and_rng(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flowtorch.visualization import plot_scalar_fields

    state = pt.get_rng_state().clone()
    with ProvenanceSession() as session:
        figure, axes = plot_scalar_fields([pt.ones(3, 4)], ["field"], 1, 1)
    assert pt.equal(state, pt.get_rng_state())
    assert len(session.records) == 1
    session.dump_provenance(tmp_path / "figure.json")
    assert (
        json.loads((tmp_path / "figure.json").read_text())["records"][0]["status"]
        == "complete"
    )
    plt.close(figure)


def test_loader_selection_snapshot_without_extra_reads():
    from flowtorch.data.dataloader import Dataloader

    class Loader(Dataloader):
        def __init__(self, association="point"):
            self._association = association
            self.reads = 0

        def load_snapshot(self, field_name, time):
            self.reads += 1
            return pt.ones(2)

        @property
        def write_times(self):
            raise AssertionError("metadata must not query times")

        @property
        def field_names(self):
            raise AssertionError("metadata must not query fields")

        @property
        def vertices(self):
            raise AssertionError("metadata must not query geometry")

        @property
        def weights(self):
            raise AssertionError("metadata must not compute weights")

    loader = Loader()
    captured = describe(loader)
    loader._association = "cell"
    assert captured["provenance"]["current"]["association"] == "point"
    assert loader.provenance["current"]["association"] == "cell"
    assert loader.reads == 0


def test_final_figure_export(tmp_path):
    import matplotlib.pyplot as plt
    from flowtorch.provenance import dump_figure_provenance

    figure, axis = plt.subplots()
    axis.set_xlim(2, 5)
    axis.set_title("updated")
    dump_figure_provenance(figure, tmp_path / "figure.json")
    record = json.loads((tmp_path / "figure.json").read_text())
    assert record["figure"]["axes"][0]["xlim"] == [2, 5]
    assert record["figure"]["axes"][0]["title"] == "updated"
    plt.close(figure)


def test_incremental_source_history_and_sharded_checkpoint(tmp_path):
    from tests.analysis.test_svd import _MatrixSource

    first = _MatrixSource(pt.arange(24, dtype=pt.float64).reshape(6, 4))
    second = _MatrixSource(pt.arange(12, dtype=pt.float64).reshape(6, 2))
    svd = SVD(first, rank=2, mode="tsqr", basis_storage="memory")
    initial_record = svd.provenance["records"][0]
    first.calls.clear()
    svd.update(second)
    assert first.calls == []
    assert svd.provenance["records"][0] == initial_record
    assert svd.provenance["records"][-1]["operation"].endswith("update")
    assert svd.provenance["records"][-1]["effective"]["cols"] == 6
    before = len(second.calls)
    svd.dump_provenance(tmp_path / "metadata.json")
    assert len(second.calls) == before
    svd.save(tmp_path / "checkpoint")
    restored = SVD.load(tmp_path / "checkpoint", basis_storage="memory")
    assert restored.provenance["records"] == svd.provenance["records"]


def test_running_moments_updates_merge_and_failed_batch(tmp_path):
    from flowtorch.analysis import RunningMoments

    data = pt.arange(18, dtype=pt.float64).reshape(3, 6)
    state = pt.get_rng_state().clone()
    moments = RunningMoments(accumulator_dtype=pt.float64)
    moments.update(data[:, :2])
    first = moments.provenance["records"][-1]
    other = RunningMoments()
    other.update(data[:, 2:])
    moments.merge(other)
    records = moments.provenance["records"]
    assert [record["operation"].split(".")[-1] for record in records] == [
        "__init__",
        "update",
        "merge",
    ]
    assert first["effective"]["count"] == 2
    assert records[-1]["effective"]["count"] == 6
    assert records[-1]["requested"]["other"]["provenance"]["current"]["count"] == 4
    fields = moments.finalize()
    assert pt.allclose(fields.mean, data.mean(dim=-1))
    assert pt.allclose(fields.variance, data.var(dim=-1, correction=0))
    assert len(moments.provenance["records"]) == 3
    assert pt.equal(state, pt.get_rng_state())
    with pytest.raises(ValueError):
        moments.update(pt.ones(2, 2, dtype=pt.float64))
    assert moments.count == 6
    assert moments.provenance["records"][-1]["status"] == "failed"
    moments.dump_provenance(tmp_path / "moments.json")
    assert json.loads((tmp_path / "moments.json").read_text())["current"]["count"] == 6


@pytest.mark.parametrize("class_name", ["MSSA", "PMSSA"])
def test_mssa_tracking_defaults_and_svd_dependencies(class_name, tmp_path):
    import flowtorch.analysis as analysis

    data = pt.arange(60, dtype=pt.float64).reshape(5, 12)
    state = pt.get_rng_state().clone()
    obj = getattr(analysis, class_name)(data, rank=2)
    assert pt.equal(state, pt.get_rng_state())
    assert obj.settings["data_matrix"]["shape"] == [5, 12]
    assert obj.settings["window_size"] is None
    records = obj.provenance["records"]
    assert len(records) == 1
    effective = records[0]["effective"]
    assert effective["window_size"] == 6
    assert effective["delay"] == 7
    assert effective["hankel_svd"]["provenance"]["current"]["rank"] == 2
    if class_name == "PMSSA":
        assert effective["projection_svd"]["provenance"]["current"]["rows"] == 5
    obj.svd.rank = 1
    assert records == obj.provenance["records"]
    obj.dump_provenance(tmp_path / "mssa.json")
    assert json.loads((tmp_path / "mssa.json").read_text())["records"] == records


def test_amps_tracking_resolved_settings(tmp_path):
    from flowtorch.analysis import AMPS

    signal = pt.sin(pt.arange(12, dtype=pt.float64))
    state = pt.get_rng_state().clone()
    obj = AMPS(signal, dt=0.1, nfft=4, adaptive=False, max_tapers=50)
    assert pt.equal(state, pt.get_rng_state())
    assert obj.settings["nfft"] == 4
    assert obj.settings["max_tapers"] == 50
    effective = obj.provenance["records"][0]["effective"]
    assert effective["nfft"] == 12
    assert effective["max_tapers"] == 12
    assert effective["signal"]["dtype"] == "torch.float64"
    obj.dump_provenance(tmp_path / "amps.json")
    assert (
        json.loads((tmp_path / "amps.json").read_text())["records"][0]["status"]
        == "complete"
    )
