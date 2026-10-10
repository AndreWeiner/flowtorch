"""Bounded, CPU-only preparation of spatial chunks."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Generator

import torch as pt
from torch.utils.data import DataLoader, Dataset


class _ChunkDataset(Dataset):
    def __init__(self, requests: list[Any], read: Callable[[Any], Any]):
        self.requests = requests
        self.read = read

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> Any:
        return self.read(self.requests[index])


def _identity(value: Any) -> Any:
    return value


def _on_cpu(value: Any) -> bool:
    if isinstance(value, pt.Tensor):
        return value.device.type == "cpu"
    if isinstance(value, dict):
        return all(_on_cpu(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_on_cpu(item) for item in value)
    return True


def iter_prepared_chunks(
    requests: Iterable[Any], read: Callable[[Any], Any], *, prefetch: bool = True
) -> Generator[Any, None, None]:
    """Prepare at most one upcoming CPU item, retaining caller-owned readers.

    DataLoader provides item dispatch, with no batching or multiprocessing.
    One background thread consumes it to avoid pickling loader handles, mesh
    arrays or resident bases. GPU-returning legacy readers stay synchronous.
    Close the iterator on early termination; shutdown drains the pending read
    before callers can delete files or close their reader.
    """
    loader = DataLoader(
        _ChunkDataset(list(requests), read),
        batch_size=None,
        num_workers=0,
        collate_fn=_identity,
        generator=pt.Generator().manual_seed(0),
    )
    iterator = iter(loader)
    sentinel = object()
    current = next(iterator, sentinel)
    if current is sentinel:
        return
    if not prefetch or not _on_cpu(current):
        yield current
        yield from iterator
        return
    grad_enabled = pt.is_grad_enabled()

    def prepare() -> Any:
        with pt.set_grad_enabled(grad_enabled):
            value = next(iterator, sentinel)
            if value is not sentinel and not _on_cpu(value):
                raise ValueError(
                    "prefetch readers must consistently return CPU tensors"
                )
            return value

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="flowtorch-read") as pool:
        while current is not sentinel:
            pending = pool.submit(prepare)
            yield current
            current = pending.result()
