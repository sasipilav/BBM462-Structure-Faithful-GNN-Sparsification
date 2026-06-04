from __future__ import annotations

from contextlib import contextmanager

import torch

from ..config import DatasetConfig
from ..types import DatasetBundle
from ..utils.graph import canonicalize_undirected_edge_index


def load_dataset(config: DatasetConfig) -> DatasetBundle:
    name = config.name.lower()
    if name in {"cora", "citeseer"}:
        return _load_planetoid(config)
    raise ValueError(f"Unsupported dataset: {config.name}")


def _load_planetoid(config: DatasetConfig) -> DatasetBundle:
    try:
        from torch_geometric.datasets import Planetoid
    except ImportError as exc:
        raise RuntimeError(f"{config.name} requires torch_geometric.") from exc

    dataset_name = "CiteSeer" if config.name.lower() == "citeseer" else "Cora"
    with _trusted_dataset_torch_load_compat():
        dataset = Planetoid(root=config.root, name=dataset_name)
    data = dataset[0]

    edge_index = canonicalize_undirected_edge_index(data.edge_index, num_nodes=data.num_nodes)
    train_idx = _mask_to_index(data.train_mask, 0)
    val_idx = _mask_to_index(data.val_mask, 0)
    test_idx = _mask_to_index(data.test_mask, 0)

    return DatasetBundle(
        name=config.name.lower(),
        x=data.x.float(),
        y=data.y.view(-1).long(),
        edge_index=edge_index,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        metadata={
            "planetoid_name": dataset_name,
            "make_undirected": config.make_undirected,
        },
    )


def _mask_to_index(mask: torch.Tensor, split_column: int) -> torch.Tensor:
    if mask.ndim == 1:
        return mask.nonzero(as_tuple=False).view(-1).long()
    if mask.ndim != 2:
        raise ValueError("Expected mask with 1 or 2 dimensions.")
    if split_column >= mask.shape[1]:
        raise IndexError(f"split_column={split_column} is out of range for mask shape {tuple(mask.shape)}")
    return mask[:, split_column].nonzero(as_tuple=False).view(-1).long()


@contextmanager
def _trusted_dataset_torch_load_compat():
    """Force weights_only=False for trusted dataset deserialization.

    PyTorch 2.6 changed torch.load(..., weights_only=True) by default.
    Some upstream dataset packages still call torch.load without explicitly
    overriding that default for pickled Data objects.
    """

    original_torch_load = torch.load
    def _compat_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            try:
                return original_torch_load(*args, **kwargs, weights_only=False)
            except TypeError as exc:
                if "weights_only" not in str(exc):
                    raise
        return original_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load
