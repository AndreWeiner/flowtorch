# third party packages
import pytest
# flowtorch packages
from flowtorch.data.utils import format_byte_size, check_and_standardize_path


def test_byte_formatting():
    size, unit = format_byte_size(1e3)
    assert size == 1e3 and unit == "b"
    size, unit = format_byte_size(1e4)
    assert size == 1e4/1024 and unit == "Kb"
    size, unit = format_byte_size(1e7)
    assert size == 1e7/1024**2 and unit == "Mb"


def test_check_and_standardize_path():
    with pytest.raises(ValueError):
        _ = check_and_standardize_path("does/not/exist/")
    path = check_and_standardize_path("./")
    assert path == "."
    path = check_and_standardize_path("./flowtorch/__init__.py", folder=False)
    assert path == "./flowtorch/__init__.py"
