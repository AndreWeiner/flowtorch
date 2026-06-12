# standard library packages
from os.path import join
import pytest
import sys
# third party packages
import torch as pt
# flowtorch packages
from flowtorch.conftest import requires_datasets


requires_dataset = requires_datasets(
    "of_cavity_ascii",
    "of_cavity_binary",
    "of_cavity_ascii_parallel",
    "of_cavity_binary_parallel",
)
pytestmark = [requires_dataset, pytest.mark.integration]

from flowtorch import FLOAT_TOLERANCE, DATASETS
from flowtorch.data import FOAMDataloader, FOAMCase, FOAMMesh


class FOAMTestData:
    def __init__(self):
        self.test_cases = [
            "of_cavity_ascii",
            "of_cavity_binary",
            "of_cavity_ascii_parallel",
            "of_cavity_binary_parallel"
        ]
        self.distributed = dict(
            zip(self.test_cases, [False, False, True, True])
        )
        self.processors = dict(
            zip(self.test_cases, [1, 1, 4, 4])
        )
        times = ["0", "0.1", "0.2", "0.3", "0.4", "0.5"]
        self.times = dict(
            zip(self.test_cases, [times] * len(self.test_cases))
        )
        self.paths = dict(
            zip(self.test_cases, [DATASETS[case] for case in self.test_cases])
        )
        self.file_paths = {}
        for case in self.test_cases:
            if self.distributed[case]:
                self.file_paths[case] = [
                    join(self.paths[case], "processor0", time, "U")
                    for time in self.times[case]
                ]
            else:
                self.file_paths[case] = [
                    join(self.paths[case], time, "U")
                    for time in self.times[case]
                ]
        field_names = {'0': ['U', 'p'],
                       '0.1': ['U', 'p', 'phi'],
                       '0.2': ['U', 'p', 'phi'],
                       '0.3': ['U', 'p', 'phi'],
                       '0.4': ['U', 'p', 'phi'],
                       '0.5': ['U', 'p', 'phi']
                       }
        self.field_names = dict(
            zip(self.test_cases, [field_names] * len(self.test_cases))
        )
        p_sum_serial = {
            "0.1": 9.02670561,
            "0.2": 8.90931304,
            "0.3": 8.90782769,
            "0.4": 8.90757253,
            "0.5": 8.90741241
        }
        p_sum_parallel = {
            "0.1": 9.02867402,
            "0.2": 8.90903473,
            "0.3": 8.90816905,
            "0.4": 8.90808059,
            "0.5": 8.90801999
        }
        U_sum_serial = {
            "0.1": [3.25148558e-01, 1.81123703e-02, 0.00000000e+00],
            "0.2": [3.25152540e-01, 1.81089523e-02, 0.00000000e+00],
            "0.3": [3.25139062e-01, 1.81036697e-02, 0.00000000e+00],
            "0.4": [3.25139344e-01, 1.81036164e-02, 0.00000000e+00],
            "0.5": [3.25139888e-01, 1.81040977e-02, 0.00000000e+00]
        }
        U_sum_parallel = {
            "0.1": [3.25149428e-01, 1.81118157e-02, 0.00000000e+00],
            "0.2": [3.25150680e-01, 1.81107100e-02, 0.00000000e+00],
            "0.3": [3.25138565e-01, 1.81028138e-02, 0.00000000e+00],
            "0.4": [3.25138639e-01, 1.81011018e-02, 0.00000000e+00],
            "0.5": [3.25138675e-01, 1.81006770e-02, 0.00000000e+00]
        }
        self.p_sum = {
            "of_cavity_ascii": p_sum_serial,
            "of_cavity_binary": p_sum_serial,
            "of_cavity_ascii_parallel": p_sum_parallel,
            "of_cavity_binary_parallel": p_sum_parallel
        }
        self.U_sum = {
            "of_cavity_ascii": U_sum_serial,
            "of_cavity_binary": U_sum_serial,
            "of_cavity_ascii_parallel": U_sum_parallel,
            "of_cavity_binary_parallel": U_sum_parallel
        }
        self.mesh_paths = {
            "of_cavity_ascii": join("constant", "polyMesh"),
            "of_cavity_binary": join("constant", "polyMesh"),
            "of_cavity_ascii_parallel": join("processor0", "constant", "polyMesh"),
            "of_cavity_binary_parallel": join("processor0", "constant", "polyMesh")
        }
        self.n_points = {
            "of_cavity_ascii": 882,
            "of_cavity_binary": 882,
            "of_cavity_ascii_parallel": 242,
            "of_cavity_binary_parallel": 242
        }
        self.n_faces = {
            "of_cavity_ascii": 1640,
            "of_cavity_binary": 1640,
            "of_cavity_ascii_parallel": 420,
            "of_cavity_binary_parallel": 420
        }
        self.first_faces = {
            "of_cavity_ascii": pt.tensor([1, 22, 463, 442], dtype=pt.int32),
            "of_cavity_binary": pt.tensor([1, 22, 463, 442], dtype=pt.int32),
            "of_cavity_ascii_parallel": pt.tensor([1, 12, 133, 122], dtype=pt.int32),
            "of_cavity_binary_parallel": pt.tensor([1, 12, 133, 122], dtype=pt.int32)
        }
        self.n_neighbors = {
            "of_cavity_ascii": 760,
            "of_cavity_binary": 760,
            "of_cavity_ascii_parallel": 180,
            "of_cavity_binary_parallel": 180
        }
        self.first_n_owners = {
            "of_cavity_ascii": pt.tensor([0, 0, 1, 1], dtype=pt.int32),
            "of_cavity_binary": pt.tensor([0, 0, 1, 1], dtype=pt.int32),
            "of_cavity_ascii_parallel": pt.tensor([0, 0, 1, 1], dtype=pt.int32),
            "of_cavity_binary_parallel": pt.tensor([0, 0, 1, 1], dtype=pt.int32)
        }
        self.first_n_neighbors = {
            "of_cavity_ascii": pt.tensor([1, 20, 2, 21], dtype=pt.int32),
            "of_cavity_binary": pt.tensor([1, 20, 2, 21], dtype=pt.int32),
            "of_cavity_ascii_parallel": pt.tensor([1, 10, 2, 11], dtype=pt.int32),
            "of_cavity_binary_parallel": pt.tensor([1, 10, 2, 11], dtype=pt.int32)
        }
        self.n_centers_volumes = {
            "of_cavity_ascii": 400,
            "of_cavity_binary": 400,
            "of_cavity_ascii_parallel": 100,
            "of_cavity_binary_parallel": 100
        }
        self.first_center = pt.tensor(
            [0.0025, 0.0025, 0.005], dtype=pt.float32)
        self.cell_volume = 2.5e-7
        self.n_cells = 400


@pytest.fixture()
def get_test_data():
    yield FOAMTestData()


class TestFOAMCase:
    def test_distributed(self, get_test_data):
        for key in get_test_data.distributed.keys():
            case = FOAMCase(get_test_data.paths[key])
            distributed = get_test_data.distributed[key]
            assert distributed == case._distributed
        for key in get_test_data.distributed.keys():
            case = FOAMCase(get_test_data.paths[key], False)
            assert case._distributed == False

    def test_processors(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            assert get_test_data.processors[key] == case._processors

    def test_build_file_path(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            file_paths_test = get_test_data.file_paths[key]
            file_paths = [
                case.build_file_path("U", time, 0)
                for time in get_test_data.times[key]
            ]
            assert file_paths_test == file_paths

    def test_write_times(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            write_times = get_test_data.times[key]
            assert write_times == case._time_folders

    def test_field_names(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            field_names = get_test_data.field_names[key]
            assert sorted(field_names) == sorted(case._field_names)

    def test_check_files(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            assert case._check_mesh_files() is True


class TestFOAMMesh:
    def test_parse_points(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            mesh = FOAMMesh(case, pt.float64)
            mesh_path = get_test_data.mesh_paths[key]
            points = mesh._parse_points(join(case._path, mesh_path))
            assert pt.sum(
                pt.abs(
                    points[0, :] - pt.Tensor([0, 0, 0])
                )
            ).item() < FLOAT_TOLERANCE
            n_points = get_test_data.n_points[key]
            assert points.size()[0] == n_points
            assert points.dtype == pt.float64

    def test_parse_faces(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            mesh = FOAMMesh(case)
            mesh_path = get_test_data.mesh_paths[key]
            n_points_faces, faces = mesh._parse_faces(join(case._path, mesh_path))
            n_faces = get_test_data.n_faces[key]
            first_face = get_test_data.first_faces[key]
            assert faces.size()[0] == n_faces
            assert pt.sum(n_points_faces - 4).item() == 0
            assert pt.sum(faces[0] - first_face).item() == 0

    def test_parse_owners_and_neighbors(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            mesh = FOAMMesh(case)
            mesh_path = get_test_data.mesh_paths[key]
            owners, neighbors = mesh._parse_owners_and_neighbors(
                join(case._path, mesh_path))
            n_owners = get_test_data.n_faces[key]
            n_neighbors = get_test_data.n_neighbors[key]
            first_owners = get_test_data.first_n_owners[key]
            first_neighbors = get_test_data.first_n_neighbors[key]
            assert owners.size()[0] == n_owners
            assert neighbors.size()[0] == n_neighbors
            assert pt.sum(
                owners[:len(first_owners)] - first_owners
            ).item() == 0
            assert pt.sum(
                neighbors[:len(first_neighbors)] - first_neighbors
            ).item() == 0

    def test_compute_cell_centers_and_volumes(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            mesh = FOAMMesh(case)
            mesh_path = get_test_data.mesh_paths[key]
            centers, volumes = mesh._compute_cell_centers_and_volumes(
                join(case._path, mesh_path))
            assert centers.size()[0] == get_test_data.n_centers_volumes[key]
            assert volumes.size()[0] == get_test_data.n_centers_volumes[key]
            assert pt.sum(
                centers[0] - get_test_data.first_center
            ).item() < FLOAT_TOLERANCE
            assert pt.sum(
                volumes - get_test_data.cell_volume
            ).item() < FLOAT_TOLERANCE

    def test_get_cell_centers(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            mesh = FOAMMesh(case)
            centers = mesh.get_cell_centers()
            volumes = mesh.get_cell_volumes()
            assert centers.size()[0] == get_test_data.n_cells
            assert volumes.size()[0] == get_test_data.n_cells
            assert pt.sum(
                centers[0] - get_test_data.first_center
            ).item() < FLOAT_TOLERANCE
            assert pt.sum(
                volumes - get_test_data.cell_volume
            ).item() < FLOAT_TOLERANCE


class TestFOAMDataloader:
    def test_load_scalar_snapshot(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = get_test_data.paths[key]
            loader = FOAMDataloader(case)
            times = loader.write_times
            p_sum = get_test_data.p_sum[key]
            for time in times[1:]:
                field = loader.load_snapshot("p", time)
                # the sum operation is more sensitive to floating point inaccuracies
                # than a direct comparison of field values; therefore, the tolerance is increased
                assert pt.abs(pt.sum(field) -
                              p_sum[time]).item() < 1.0e-4

    def test_load_vector_snapshot(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = get_test_data.paths[key]
            loader = FOAMDataloader(case)
            times = loader.write_times
            U_sum = get_test_data.U_sum[key]
            for time in times[1:]:
                field = loader.load_snapshot("U", time)
                difference = pt.sum(
                    pt.abs(pt.sum(field, dim=0) - pt.tensor(U_sum[time])))
                # the sum operation is more sensitive to floating point inaccuracies
                # than a direct comparison of field values; therefore, the tolerance is increased
                assert difference.item() < 1.0e-4

    def test_load_unsupported_field(self, get_test_data):
        key = list(get_test_data.paths.keys())[0]
        loader = FOAMDataloader(get_test_data.paths[key])
        times = loader.write_times
        with pytest.raises(ValueError):
            loader.load_snapshot("phi", times[-1])

    def test_load_multiple_fields(self, get_test_data):
        key = list(get_test_data.paths.keys())[0]
        loader = FOAMDataloader(get_test_data.paths[key])
        times = loader.write_times
        data = loader.load_snapshot(["p", "U"], times[-1])
        assert len(data) == 2

    def test_load_multiple_times(self, get_test_data):
        key = list(get_test_data.paths.keys())[0]
        loader = FOAMDataloader(get_test_data.paths[key])
        times = loader.write_times[1:]
        data = loader.load_snapshot("U", times)
        assert data.shape[-1] == len(times)
        assert data.shape[-2] == 3
