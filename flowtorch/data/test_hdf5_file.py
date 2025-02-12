# standard library packages
import os
import pytest
# third party packages
import torch as pt
from h5py import File
# flowtorch packages
from flowtorch import DATASET_PATH, DATASETS
from flowtorch.data import (HDF5Dataloader, HDF5Writer, FOAM2HDF5,
                            XDMFWriter, copy_hdf5_mesh)


class HDF5TestData:
    def __init__(self):
        self.test_cases = [
            "of_cavity_ascii",
            "of_cavity_binary",
            "of_cavity_ascii_parallel",
            "of_cavity_binary_parallel"
        ]
        self.const_group = sorted(
            ["vertices", "connectivity", "centers", "volumes"])
        self.var_group = ["0.1", "0.2", "0.3", "0.4", "0.5"]


@pytest.fixture()
def get_test_data():
    yield HDF5TestData()


class TestHDF5Writer:
    def test_write(self, get_test_data):
        file_path = DATASET_PATH + "test_file.hdf5"
        writer = HDF5Writer(file_path)
        writer.write("ones", (3, 2), pt.ones(3, 2), "0.01")
        writer.write("twos", (3, 2), pt.ones(3, 2)*2, "0.01")
        writer.write("threes", (3, 2), pt.ones(3, 2)*3, "0.03")
        del writer
        hdf5_file = File(file_path, mode="a")
        assert os.path.isfile(file_path)
        assert list(hdf5_file["variable"].keys()) == ["0.01", "0.03"]
        hdf5_file.close()
        os.remove(file_path)

    def test_write_const(self, get_test_data):
        file_path = DATASET_PATH + "test_file.hdf5"
        writer = HDF5Writer(file_path)
        writer.write("zeros", (3, 2), pt.zeros(
            (3, 2), dtype=pt.float64), dtype=pt.float64)
        writer.write("zeros_single", (3, 2), pt.zeros(
            (3, 2), dtype=pt.float32), dtype=pt.float32)
        writer.write("zeros_int", (3, 2), pt.zeros(
            (3, 2), dtype=pt.int32), dtype=pt.int32)
        del writer
        hdf5_file = File(file_path, mode="a")
        assert list(hdf5_file["constant"].keys()) == [
            "zeros", "zeros_int", "zeros_single"]
        assert hdf5_file["constant/zeros"].dtype == "float64"
        assert hdf5_file["constant/zeros_single"].dtype == "float32"
        assert hdf5_file["constant/zeros_int"].dtype == "int32"
        hdf5_file.close()
        os.remove(file_path)

    def test_write_xdmf(self, get_test_data):
        case = get_test_data.test_cases[0]
        case_path = DATASET_PATH + case
        converter = FOAM2HDF5(case_path)
        converter.convert("flowtorch.hdf5", None, ["0.5"])
        del converter
        file_path = case_path + "/flowtorch.hdf5"
        writer = XDMFWriter.from_filepath(file_path)
        writer.create_xdmf()
        assert os.path.isfile(case_path + "/flowtorch.xdmf")
        os.remove(file_path)
        os.remove(case_path + "/flowtorch.xdmf")


def test_conversion(get_test_data):
    for case in get_test_data.test_cases[:2]:
        case_path = DATASET_PATH + case
        converter = FOAM2HDF5(case_path)
        # test all fields, selected times
        converter.convert("flowtorch.hdf5", None,
                          ["0.1", "0.2", "0.3", "0.4", "0.5"])
        del converter
        filename = case_path + "/flowtorch.hdf5"
        if os.path.isfile(filename):
            hdf5_file = File(filename, mode="a")
            const_keys = sorted(hdf5_file["constant"].keys())
            assert const_keys == get_test_data.const_group
            var_keys = sorted(hdf5_file["variable"].keys())
            assert var_keys == get_test_data.var_group
            assert hdf5_file["constant/volumes"].shape[0] == 400
            assert hdf5_file["constant/centers"].shape == (400, 3)
            hdf5_file.close()
            os.remove(filename)
            os.remove(case_path + "/flowtorch.xdmf")
        # test selected field, selected times
        converter = FOAM2HDF5(case_path)
        converter.convert("flowtorch.hdf5", ["U"], ["0.1", "0.2"])
        del converter
        if os.path.isfile(filename):
            hdf5_file = File(filename, mode="a")
            var_keys = sorted(hdf5_file["variable"].keys())
            assert var_keys == ["0.1", "0.2"]
            assert list(hdf5_file["variable/0.1"].keys()) == ["U"]
            hdf5_file.close()
            os.remove(filename)
            os.remove(case_path + "/flowtorch.xdmf")


def test_hdf5_dataloader():
    path = DATASETS["of_cavity_ascii"]
    converter = FOAM2HDF5(path)
    converter.convert("flowtorch.hdf5", None,
                      ["0.1", "0.2", "0.3", "0.4", "0.5"])
    file_path = path + "/flowtorch.hdf5"
    loader = HDF5Dataloader(file_path)
    times = loader.write_times
    assert times[-1] == "0.5"
    assert len(times) == 5
    fields = loader.field_names
    assert sorted(fields["0.5"]) == sorted(["p", "U"])
    vertices = loader.vertices
    assert vertices.shape == (400, 3)
    weights = loader.weights
    assert weights.shape == (400,)
    # load single snapshot, single field
    p = loader.load_snapshot("p", times[0])
    assert p.shape == (400,)
    # load multiple snapshots, single field
    p_series = loader.load_snapshot("p", times)
    assert p_series.shape[-1] == len(times)
    assert pt.allclose(p_series[:, 0], p)
    # load single snapshot, multiple fields
    p, U = loader.load_snapshot(["p", "U"], times[0])
    assert p.shape == (400,)
    assert U.shape == (400, 3)
    # load multiple snapshots, multiple fields
    ps, Us = loader.load_snapshot(["p", "U"], times)
    assert ps.shape[-1] == len(times)
    assert Us.shape[-1] == len(times)
    assert pt.allclose(ps[:, 0], p)
    assert pt.allclose(Us[:, :, 0], U)
    os.remove(file_path)
    os.remove(path + "/flowtorch.xdmf")


def test_copy_hdf5_mesh(get_test_data):
    for case in get_test_data.test_cases[:2]:
        case_path = DATASET_PATH + case
        converter = FOAM2HDF5(case_path)
        converter.convert("flowtorch.hdf5", ["U"], ["0.1", "0.2"])
        del converter
        if os.path.isfile(os.path.join(case_path, "flowtorch.hdf5")):
            copy_hdf5_mesh(case_path, "flowtorch.hdf5", "mesh_only.hdf5")
            assert os.path.isfile(os.path.join(case_path, "mesh_only.hdf5"))
            assert os.path.isfile(os.path.join(case_path, "mesh_only.xdmf"))
            hdf5_file = File(os.path.join(
                case_path, "mesh_only.hdf5"), mode="a")
            assert hdf5_file["constant/volumes"].shape[0] == 400
            assert hdf5_file["constant/centers"].shape == (400, 3)
            assert "constant/vertices" in hdf5_file
            assert "constant/connectivity" in hdf5_file
            hdf5_file.close()
            for f in ("flowtorch.hdf5", "flowtorch.xdmf", "mesh_only.hdf5", "mesh_only.xdmf"):
                os.remove(os.path.join(case_path, f))
