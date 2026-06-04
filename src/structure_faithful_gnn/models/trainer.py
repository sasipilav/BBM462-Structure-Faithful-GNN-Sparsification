from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from ..config import ModelConfig
from ..types import DatasetBundle
from ..utils.seed import set_seed


def _load_pyg_layers() -> tuple[type[nn.Module], type[nn.Module]]:
    try:
        from torch_geometric.nn import GCNConv, SAGEConv
    except ImportError as exc:
        raise RuntimeError(
            "torch_geometric is required by this project. Install PyG before running dense or pruning experiments."
        ) from exc
    return GCNConv, SAGEConv


@dataclass
class TrainingResult:
    accuracy: float
    macro_f1: float
    train_sec: float
    infer_sec: float
    best_val_accuracy: float
    best_epoch: int
    model_name: str
    backend: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "train_sec": self.train_sec,
            "infer_sec": self.infer_sec,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "model_name": self.model_name,
            "backend": self.backend,
        }


class PyGGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        gcn_conv, _ = _load_pyg_layers()
        dims = [input_dim] + [hidden_dim] * max(num_layers - 1, 1) + [output_dim]
        self.layers = nn.ModuleList(
            [gcn_conv(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)]
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight=edge_weight)
            if idx != len(self.layers) - 1:
                h = torch.relu(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return h


class PyGGraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        _, sage_conv = _load_pyg_layers()
        dims = [input_dim] + [hidden_dim] * max(num_layers - 1, 1) + [output_dim]
        self.layers = nn.ModuleList(
            [sage_conv(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)]
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_weight
        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if idx != len(self.layers) - 1:
                h = torch.relu(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return h


def build_model(config: ModelConfig, input_dim: int, output_dim: int) -> tuple[nn.Module, str]:
    name = config.name.lower()
    if name == "gcn":
        return PyGGCN(input_dim, config.hidden_dim, output_dim, config.dropout, config.num_layers), "pyg"
    if name == "graphsage":
        return PyGGraphSAGE(input_dim, config.hidden_dim, output_dim, config.dropout, config.num_layers), "pyg"
    raise ValueError(f"Unsupported model: {config.name}")


def train_and_evaluate(bundle: DatasetBundle, config: ModelConfig, seed: int) -> TrainingResult:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = bundle.x.to(device)
    y = bundle.y.to(device)
    edge_index = bundle.model_edge_index().to(device)
    edge_weight = bundle.model_edge_weight()
    edge_weight = edge_weight.to(device) if edge_weight is not None else None
    model_edge_index, model_edge_weight = _prepare_model_graph(
        config,
        edge_index,
        edge_weight,
        num_nodes=bundle.num_nodes,
    )
    train_idx = bundle.train_idx.to(device)
    val_idx = bundle.val_idx.to(device)
    test_idx = bundle.test_idx.to(device)

    model, backend = build_model(config, bundle.num_features, bundle.num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    best_epoch = 0
    bad_epochs = 0

    train_start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, model_edge_index, model_edge_weight)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        val_accuracy = _accuracy_from_logits(logits[val_idx], y[val_idx])
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.early_stop:
            break

    train_sec = time.perf_counter() - train_start

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        infer_start = time.perf_counter()
        logits = model(x, model_edge_index, model_edge_weight)
        infer_sec = time.perf_counter() - infer_start
        preds = logits[test_idx].argmax(dim=1).cpu().numpy()
        targets = y[test_idx].cpu().numpy()

    return TrainingResult(
        accuracy=float(accuracy_score(targets, preds)),
        macro_f1=float(f1_score(targets, preds, average="macro")),
        train_sec=float(train_sec),
        infer_sec=float(infer_sec),
        best_val_accuracy=float(best_val_accuracy),
        best_epoch=int(best_epoch),
        model_name=config.name,
        backend=backend,
    )


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    return float(accuracy_score(labels.detach().cpu().numpy(), preds))


def _prepare_model_graph(
    config: ModelConfig,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    *,
    num_nodes: int,
) -> tuple[torch.Tensor | object, torch.Tensor | None]:
    if edge_weight is None or config.name.lower() != "graphsage":
        return edge_index, edge_weight
    try:
        import torch_scatter  # noqa: F401
        from torch_sparse import SparseTensor
    except ImportError as exc:
        raise RuntimeError(
            "Weighted GraphSAGE training requires both torch_scatter and torch_sparse. "
            "Install the matching PyG companion wheels from https://data.pyg.org for the active torch/cuda build "
            "before running GraphSAGE on weighted artifacts such as DSpar."
        ) from exc
    adj_t = SparseTensor(
        row=edge_index[1],
        col=edge_index[0],
        value=edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )
    return adj_t, None
