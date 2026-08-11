# third party libraries
import numpy as np
from pytest import raises

# flowtorch packages
import flowtorch.rom.utils as utils_module
from flowtorch.rom.utils import (
    log_time,
    check_int_larger_than,
    remove_sequential_duplicates,
)


def test_log_time(monkeypatch):
    timestamps = iter((10.0, 10.1))
    monkeypatch.setattr(utils_module, "time", lambda: next(timestamps))

    @log_time
    def operation():
        return {"test": 0}

    log = operation()
    assert "execution_time" in log.keys()
    assert "test" in log.keys()
    assert np.isclose(log["execution_time"], 0.1)


def test_check_int_larger_than():
    with raises(ValueError):
        check_int_larger_than(1.0, 0, "name")
    with raises(ValueError):
        check_int_larger_than(0, 0, "name")
    check_int_larger_than(1, 0, "name")


def test_remove_sequential_duplicates():
    sequence = np.array([1, 1, 2, 2, 3, 4, 5, 5])
    assert np.allclose(
        remove_sequential_duplicates(sequence), np.array([1, 2, 3, 4, 5])
    )
