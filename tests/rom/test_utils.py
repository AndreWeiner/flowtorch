# standard library packages
from time import sleep
from pytest import raises
# third party libraries
import numpy as np
# flowtorch packages
from flowtorch.rom.utils import (log_time, check_int_larger_than,
                                 remove_sequential_duplicates)


def test_log_time():
    @log_time
    def wait_seconds(time_to_wait: float):
        sleep(time_to_wait)
        return {"test": 0}
    log = wait_seconds(0.1)
    assert "execution_time" in log.keys()
    assert "test" in log.keys()
    assert abs(log["execution_time"] - 0.1) < 0.01


def test_check_int_larger_than():
    with raises(ValueError):
        check_int_larger_than(1.0, 0, "name")
    with raises(ValueError):
        check_int_larger_than(0, 0, "name")
    check_int_larger_than(1, 0, "name")


def test_remove_sequential_duplicates():
    sequence = np.array([1, 1, 2, 2, 3, 4, 5, 5])
    assert np.allclose(
        remove_sequential_duplicates(sequence),
        np.array([1, 2, 3, 4, 5])
    )
