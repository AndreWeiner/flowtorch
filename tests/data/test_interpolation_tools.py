import pytest
import torch as pt

from flowtorch.data import replace_masked_values


def _grid():
    return pt.meshgrid(pt.arange(5.0), pt.arange(5.0), indexing="ij")


def test_replace_masked_values_interpolates_single_field():
    x, y = _grid()
    field = x + 2.0 * y
    mask = pt.ones((5, 5), dtype=pt.bool)
    mask[2, 2] = False

    repaired = replace_masked_values(x, y, field, mask)

    assert repaired.shape == field.shape
    assert repaired[2, 2] == pytest.approx(field[2, 2].item())
    assert pt.equal(repaired[mask], field[mask])


def test_replace_masked_values_interpolates_sequence():
    x, y = _grid()
    field = pt.stack((x + y, 2.0 * x - y), dim=-1)
    mask = pt.ones((5, 5), dtype=pt.bool)
    mask[2, 2] = False

    repaired = replace_masked_values(x, y, field, mask)

    assert repaired.shape == field.shape
    assert pt.allclose(repaired[2, 2], field[2, 2])


def test_replace_masked_values_returns_copy_for_complete_mask():
    x, y = _grid()
    field = x + y
    mask = pt.ones((5, 5), dtype=pt.bool)

    repaired = replace_masked_values(x, y, field, mask)

    assert repaired is not field
    assert pt.equal(repaired, field)


def test_replace_masked_values_rejects_invalid_inputs():
    x, y = _grid()
    field = x + y
    mask = pt.ones((5, 5), dtype=pt.bool)

    with pytest.raises(ValueError):
        replace_masked_values(x[:-1], y, field, mask)
    with pytest.raises(ValueError):
        replace_masked_values(x, y, field, pt.zeros_like(mask))
