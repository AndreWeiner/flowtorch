"""Tests for lazy physical state-vector construction."""

import pytest
import torch as pt
from h5py import File

from flowtorch.analysis.svd import SVD
from flowtorch.analysis.state_vector import (
    DataloaderStateVectorSource,
    FieldSpec,
    StateVectorLayout,
    StateVectorResult,
)
from flowtorch.data.dataloader import Dataloader
from flowtorch.data.hdf5_file import HDF5Dataloader


class _Loader(Dataloader):
    def __init__(self):
        time = pt.arange(4, dtype=pt.float64)
        self.data = {
            "p": pt.arange(5, dtype=pt.float64)[:, None] + time,
            "U": pt.stack(
                (
                    pt.arange(5, dtype=pt.float64)[:, None] + time,
                    10.0 + time.expand(5, -1),
                ),
                dim=1,
            ),
        }
        self.slice_calls = []

    def load_snapshot(self, field_name, time):
        fields = field_name if isinstance(field_name, list) else [field_name]
        times = time if isinstance(time, list) else [time]
        indices = [int(value) for value in times]
        loaded = [self.data[name][..., indices] for name in fields]
        if not isinstance(time, list):
            loaded = [value[..., 0] for value in loaded]
        return loaded if isinstance(field_name, list) else loaded[0]

    def load_snapshot_slice(self, field_name, time, spatial_slice):
        self.slice_calls.append(spatial_slice)
        loaded = self.load_snapshot(field_name, time)
        if isinstance(loaded, list):
            return [value[spatial_slice] for value in loaded]
        return loaded[spatial_slice]

    def snapshot_shape(self, field_name, time):
        return self.data[field_name].shape[:-1]

    @property
    def write_times(self):
        return [str(index) for index in range(4)]

    @property
    def field_names(self):
        return {time: ["p", "U"] for time in self.write_times}

    @property
    def vertices(self):
        return pt.arange(5)[:, None]

    @property
    def weights(self):
        return pt.ones(5)


def test_layout_uses_one_factor_per_field_and_round_trips_components():
    layout = StateVectorLayout(
        (FieldSpec("U", (2,), ("u", "v")), FieldSpec("p")),
        (3,),
        {"U": 2.0, "p": 10.0},
    )
    fields = {
        "U": pt.arange(18, dtype=pt.float64).reshape(3, 2, 3),
        "p": pt.arange(9, dtype=pt.float64).reshape(3, 3),
    }

    state = layout.pack(fields)
    assert state.shape == (9, 3)
    pt.testing.assert_close(state[:3], fields["U"][:, 0] / 2.0)
    pt.testing.assert_close(state[3:6], fields["U"][:, 1] / 2.0)
    restored = layout.split(state)
    pt.testing.assert_close(restored["U"], fields["U"])
    pt.testing.assert_close(restored["p"], fields["p"])


def test_layout_round_trips_tensor_components():
    layout = StateVectorLayout((FieldSpec("stress", (2, 2)),), (3,))
    field = pt.arange(36, dtype=pt.float64).reshape(3, 2, 2, 3)

    restored = layout.split(layout.pack({"stress": field}))

    pt.testing.assert_close(restored["stress"], field)


def test_normalization_requires_exactly_one_scalar_per_field():
    fields = (FieldSpec("U", (2,)), FieldSpec("p"))

    with pytest.raises(ValueError, match="exactly one factor per field"):
        StateVectorLayout(fields, (3,), {})
    with pytest.raises(ValueError, match="real scalars"):
        StateVectorLayout(fields, (3,), {"U": [1.0, 2.0], "p": 1.0})


def test_result_keeps_flat_values_normalized_and_split_values_physical():
    layout = StateVectorLayout(
        (FieldSpec("U", (2,)), FieldSpec("p")),
        (4,),
        {"U": 2.0, "p": 5.0},
    )
    physical = {
        "U": pt.arange(8, dtype=pt.float64).reshape(4, 2),
        "p": pt.arange(4, dtype=pt.float64),
    }

    def produce(spatial_slice):
        shape = (spatial_slice.stop - spatial_slice.start,)
        values = {name: value[spatial_slice] for name, value in physical.items()}
        return layout.pack(values, spatial_shape=shape)

    result = StateVectorResult(layout, slice(0, 4), (), produce, 2)

    pt.testing.assert_close(result.materialize_local(), layout.pack(physical))
    split = result.materialize_local(split=True)
    pt.testing.assert_close(split["U"], physical["U"])
    pt.testing.assert_close(split["p"], physical["p"])


def test_dataloader_source_reads_only_requested_chunks():
    loader = _Loader()
    source = DataloaderStateVectorSource(
        loader,
        (FieldSpec("U", (2,), ("u", "v")), FieldSpec("p")),
        normalization={"U": 2.0, "p": 4.0},
    )

    state = source.read(slice(1, 4), slice(1, 3))

    assert state.shape == (9, 2)
    assert loader.slice_calls == [slice(1, 4)]
    restored = source.layout.split(state, spatial_shape=(3,))
    pt.testing.assert_close(restored["U"], loader.data["U"][1:4, :, 1:3])
    pt.testing.assert_close(restored["p"], loader.data["p"][1:4, 1:3])


def test_fitted_rms_uses_one_vector_magnitude_factor():
    loader = _Loader()
    source = DataloaderStateVectorSource(
        loader,
        (FieldSpec("U", (2,), ("u", "v")), FieldSpec("p")),
        normalization="rms",
        spatial_weight=loader.weights,
    )

    source.fit_normalization(slice(0, 5), 2, 2)

    velocity = loader.data["U"]
    pressure = loader.data["p"]
    velocity_fluctuation = velocity - velocity.mean(dim=-1, keepdim=True)
    pressure_fluctuation = pressure - pressure.mean(dim=-1, keepdim=True)
    expected_velocity = velocity_fluctuation.square().sum() / (4 * 5)
    expected_pressure = pressure_fluctuation.square().sum() / (4 * 5)
    factors = source.layout.normalization_factors
    pt.testing.assert_close(
        pt.tensor(factors["U"] ** 2, dtype=pt.float64), expected_velocity
    )
    pt.testing.assert_close(
        pt.tensor(factors["p"] ** 2, dtype=pt.float64), expected_pressure
    )


def test_checkpoint_restores_fitted_rms_without_scanning_snapshots(tmp_path):
    fields = (FieldSpec("U", (2,), ("u", "v")), FieldSpec("p"))
    source = DataloaderStateVectorSource(_Loader(), fields, normalization="rms")
    svd = SVD(source, rank=3, spatial_batch_size=2, snapshot_batch_size=2)
    path = tmp_path / "fitted-rms-svd.pt"
    svd.save(path)
    restored_loader = _Loader()
    restored_source = DataloaderStateVectorSource(
        restored_loader, fields, normalization="rms"
    )

    restored = SVD.load(path, source=restored_source)

    assert restored_loader.slice_calls == []
    assert restored_source.normalization_fitted
    assert (
        restored_source.layout.normalization_factors
        == source.layout.normalization_factors
    )
    pt.testing.assert_close(restored.s, svd.s)


def test_hdf5_loader_reads_spatial_slice_directly(tmp_path):
    path = tmp_path / "fields.h5"
    with File(path, "w") as output:
        variables = output.create_group("variable")
        for index, time in enumerate(("0", "1")):
            group = variables.create_group(time)
            group.create_dataset("p", data=(pt.arange(6) + index).numpy().reshape(6, 1))
            group.create_dataset(
                "U", data=(pt.arange(18) + index).numpy().reshape(6, 3)
            )
        constants = output.create_group("constant")
        constants.create_dataset("centers", data=pt.zeros((6, 3)).numpy())
        constants.create_dataset("volumes", data=pt.ones(6).numpy())

    loader = HDF5Dataloader(str(path), dtype=pt.float64)
    pressure, velocity = loader.load_snapshot_slice(["p", "U"], ["0", "1"], slice(2, 5))

    assert pressure.shape == (3, 2)
    assert velocity.shape == (3, 3, 2)
    assert loader.snapshot_shape("p", "0") == (6,)
    assert loader.snapshot_shape("U", "0") == (6, 3)
