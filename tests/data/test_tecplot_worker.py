"""Unit tests for composite traversal in the ParaView-side worker."""

import numpy as np
import pytest

from flowtorch.data._tecplot_pvpython import (
    _copy_field,
    _field_association,
    _handle,
    _leaf_records,
    _resolve_leaf,
)

_NAME_KEY = object()


class _Metadata:
    def __init__(self, name=None, text="metadata without a name"):
        self.name = name
        self.text = text

    def Has(self, key):
        return key is _NAME_KEY and self.name is not None

    def Get(self, key):
        assert key is _NAME_KEY
        return self.name

    def __str__(self):
        return self.text


class _Composite:
    def __init__(self, children, names=None, metadata=None):
        self.children = children
        self.names = names or [None] * len(children)
        self.metadata = metadata

    def GetNumberOfBlocks(self):
        return len(self.children)

    def GetBlock(self, index):
        return self.children[index]

    def GetMetaData(self, index):
        if self.metadata is not None:
            return self.metadata[index]
        return _Metadata(self.names[index])


class _Leaf:
    def __init__(self, points=3, cells=2):
        self.points = points
        self.cells = cells

    def GetNumberOfPoints(self):
        return self.points

    def GetNumberOfCells(self):
        return self.cells


class _Wrapper:
    def __init__(self, leaf, point_data=None, cell_data=None):
        self.PointData = point_data or {}
        self.CellData = cell_data or {}
        self.Points = np.arange(leaf.points * 3, dtype=float).reshape(leaf.points, 3)


class _Reader:
    datasets = {}

    def __init__(self, registrationName, FileName):
        self.registration_name = registrationName
        self.file_name = FileName[0]
        self.information_updated = False
        self.pipeline_updated = False

    def UpdatePipelineInformation(self):
        self.information_updated = True

    def UpdatePipeline(self):
        self.pipeline_updated = True


class _ServerManager:
    @staticmethod
    def Fetch(reader):
        assert reader.information_updated
        assert reader.pipeline_updated
        return _Reader.datasets[reader.file_name]


class _DatasetAdapter:
    wrappers = {}

    @classmethod
    def WrapDataObject(cls, leaf):
        return cls.wrappers[leaf]


def _request(operation, **parameters):
    return {
        "operation": operation,
        "registration_name": "synthetic",
        "file_path": "synthetic.plt",
        **parameters,
    }


def test_flat_leaf_names_remain_short_and_paths_are_stable():
    lower = _Leaf()
    upper = _Leaf()
    root = _Composite([lower, upper], names=["Lower", "Upper"])

    assert _leaf_records(root, _NAME_KEY) == [
        {"name": "Lower", "full_name": "Lower", "path": [0]},
        {"name": "Upper", "full_name": "Upper", "path": [1]},
    ]
    assert _resolve_leaf(root, [1]) is upper


def test_nested_duplicate_leaf_names_use_full_paths():
    left_wing = _Leaf()
    right_wing = _Leaf()
    left = _Composite([left_wing], names=["WingUpper"])
    right = _Composite([right_wing], names=["WingUpper"])
    root = _Composite([left, right], names=["Left", "Right"])

    assert _leaf_records(root, _NAME_KEY) == [
        {
            "name": "Left/WingUpper",
            "full_name": "Left/WingUpper",
            "path": [0, 0],
        },
        {
            "name": "Right/WingUpper",
            "full_name": "Right/WingUpper",
            "path": [1, 0],
        },
    ]


def test_metadata_text_fallback_and_deterministic_fallback_name():
    parsed = _Leaf()
    unnamed = _Leaf()
    root = _Composite(
        [parsed, unnamed],
        metadata=[
            _Metadata(text="vtkInformation\n  NAME: ParsedName"),
            _Metadata(),
        ],
    )

    assert [record["name"] for record in _leaf_records(root, _NAME_KEY)] == [
        "ParsedName",
        "block_1",
    ]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ([2], "has no child"),
        ([0], "non-leaf"),
        ([0, 1], "empty block"),
        ([1, 0], "continues past leaf"),
        ("0/1", "Invalid block path"),
    ],
)
def test_resolve_leaf_reports_invalid_selections(path, message):
    root = _Composite([_Composite([_Leaf(), None]), _Leaf()], names=["group", "leaf"])
    with pytest.raises(ValueError, match=message):
        _resolve_leaf(root, path)


def test_fields_are_selected_by_name_and_association():
    leaf = _Leaf(points=3, cells=2)
    wrapper = _Wrapper(
        leaf,
        point_data={"pressure": np.array([1.0, 2.0, 3.0])},
        cell_data={"volume": np.array([4.0, 5.0])},
    )

    assert np.array_equal(
        _copy_field(leaf, wrapper, "pressure", "point", np),
        np.array([1.0, 2.0, 3.0]),
    )
    assert np.array_equal(
        _copy_field(leaf, wrapper, "volume", "cell", np), np.array([4.0, 5.0])
    )
    with pytest.raises(ValueError, match="missing from point data"):
        _copy_field(leaf, wrapper, "volume", "point", np)


def test_legacy_field_request_rejects_ambiguous_association():
    leaf = _Leaf()
    wrapper = _Wrapper(
        leaf,
        point_data={"quality": np.arange(3)},
        cell_data={"quality": np.arange(2)},
    )
    with pytest.raises(ValueError, match="both point and cell data"):
        _field_association(wrapper, "quality")


def test_field_length_must_match_selected_leaf():
    leaf = _Leaf(points=3)
    wrapper = _Wrapper(leaf, point_data={"pressure": np.arange(2)})
    with pytest.raises(ValueError, match="has 2 values.*3 points"):
        _copy_field(leaf, wrapper, "pressure", "point", np)


def test_handle_loads_zone_specific_fields_and_vertices():
    leaf = _Leaf(points=3)
    root = _Composite([_Composite([leaf], names=["WingUpper"])], names=["root"])
    wrapper = _Wrapper(
        leaf,
        point_data={"cp": np.array([0.1, 0.2, 0.3])},
        cell_data={"area": np.array([1.0, 2.0])},
    )
    _Reader.datasets["synthetic.plt"] = root
    _DatasetAdapter.wrappers[leaf] = wrapper
    deleted = []

    kind, names = _handle(
        _request("field_names", block_path=[0, 0], association="point"),
        _Reader,
        _ServerManager,
        _DatasetAdapter,
        deleted.append,
        np,
        _NAME_KEY,
    )
    assert kind == "json"
    assert names == ["cp"]

    kind, field = _handle(
        _request("field", block_path=[0, 0], field_name="area", association="cell"),
        _Reader,
        _ServerManager,
        _DatasetAdapter,
        deleted.append,
        np,
        _NAME_KEY,
    )
    assert kind == "array"
    assert np.array_equal(field, np.array([1.0, 2.0]))

    kind, vertices = _handle(
        _request("vertices", block_path=[0, 0]),
        _Reader,
        _ServerManager,
        _DatasetAdapter,
        deleted.append,
        np,
        _NAME_KEY,
    )
    assert kind == "array"
    assert np.array_equal(vertices, wrapper.Points)


def test_handle_rejects_empty_geometry():
    leaf = _Leaf(points=0, cells=0)
    root = _Composite([leaf], names=["empty"])
    wrapper = _Wrapper(leaf)
    _Reader.datasets["synthetic.plt"] = root
    _DatasetAdapter.wrappers[leaf] = wrapper

    with pytest.raises(ValueError, match="empty geometry"):
        _handle(
            _request("vertices", block_path=[0]),
            _Reader,
            _ServerManager,
            _DatasetAdapter,
            lambda reader: None,
            np,
            _NAME_KEY,
        )
