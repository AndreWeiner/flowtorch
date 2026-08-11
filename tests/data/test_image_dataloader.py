"""Unittests for the ImageDataloader class."""

# standard library packages
from os.path import join

# third party packages
from pytest import raises, fixture
from PIL import Image
import torch as pt
import numpy as np

# flowTorch packages
from flowtorch.data.image_dataloader import ImageDataloader


@fixture()
def create_image_dir(tmp_path):
    H, W = 12, 11
    prefix = ""
    data = {
        "0001": 0,
        "0002": 2**16 // 2,
        "0003": 2**16 - 1,
    }
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for t, v in data.items():
        img = np.full((H, W), v, dtype=np.uint16)
        Image.fromarray(img).save(image_dir / f"{prefix}{t}.tif")
    return str(image_dir), data, prefix


def test_ImageDataloader(create_image_dir):
    path, data, prefix = create_image_dir
    loader = ImageDataloader(path, prefix, dtype=pt.float32)
    assert loader.write_times == list(data.keys())
    assert join(path, f"{prefix}0001.tif") == loader._build_file_path("0001")
    img = loader.load_snapshot(loader.write_times[-1])
    assert img.dtype == pt.float32
    assert img.shape == (12, 11)
    assert pt.allclose(pt.full(img.shape, 1.0, dtype=img.dtype), img)
    img_seq = loader.load_snapshot(loader.write_times)
    assert img_seq.shape == (12, 11, len(data.keys()))
    assert pt.all(img_seq <= 1.0)
    with raises(NotImplementedError):
        _ = loader.field_names
    with raises(NotImplementedError):
        _ = loader.vertices
    with raises(NotImplementedError):
        _ = loader.weights
