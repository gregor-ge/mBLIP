from typing import Optional, cast

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset

from trident.core.trident import TridentModule
from trident.utils.logging import get_logger

log = get_logger(__name__)


def get_module():
    network = nn.Linear(10, 1, bias=False)
    network.train()
    network.weight = nn.Parameter(torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    return network


def get_val_data(from_: Optional[int] = None, to_: Optional[int] = None):
    examples = None
    if isinstance(from_, int):
        examples = torch.zeros(10 - from_, 10)
        for row, col in enumerate(range(from_, 10)):
            examples[row, col] = 1
    if isinstance(to_, int):
        examples = torch.zeros(to_, 10)
        for row, col in enumerate(range(to_)):
            examples[row, col] = 1
    assert isinstance(examples, torch.Tensor)
    return examples


class ToyModule(TridentModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def batch_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ret = {}
        ret["preds"] = self.model(batch["examples"])
        if (labels := batch.get("labels", None)) is not None:
            assert isinstance(labels, torch.Tensor)  # satisfy linter
            ret["loss"] = F.mse_loss(ret["preds"], labels)
        return ret

    def forward(
        self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        if "examples" in batch:
            b = cast(dict[str, torch.Tensor], batch)
            return self.batch_forward(b)
        else:
            b = cast(dict[str, dict[str, torch.Tensor]], batch)
            # runs only with multi train dataset
            first_half_correct = b["first_half"]["examples"].sum(0)[:5].sum() == 5
            second_half_correct = b["second_half"]["examples"].sum(0)[5:].sum() == 5
            assert first_half_correct.item(), "First half has incorrect examples"
            assert second_half_correct.item(), "Second half has incorrect examples"
            rets = {
                dataset_name: self.batch_forward(dataset_batch)
                for dataset_name, dataset_batch in b.items()
            }
            loss = torch.stack([v["loss"] for v in rets.values()]).mean()
            return {"loss": loss}


class IdentityDataset(Dataset):
    def __init__(
        self, X: torch.Tensor | DictConfig, y: torch.Tensor | DictConfig
    ) -> None:
        super().__init__()
        self.X: torch.Tensor = (
            X if isinstance(X, torch.Tensor) else hydra.utils.instantiate(X)
        )
        self.y: torch.Tensor = (
            y if isinstance(y, torch.Tensor) else hydra.utils.instantiate(y)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i])


def collate_fn(
    examples: list[tuple[torch.Tensor, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    X, y = zip(*examples)
    batch = {"examples": torch.vstack(X), "labels": torch.stack(y)}
    return batch


def simple_test():
    network = nn.Linear(10, 1, bias=False)
    network.train()
    network.weight = nn.Parameter(torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    optim = torch.optim.SGD(network.parameters(), lr=5)
    dataset = IdentityDataset(
        torch.eye(10), torch.Tensor.float(torch.arange(start=1, end=11))
    )

    for _ in range(10):
        X, y = dataset.X, dataset.y
        preds = network(X)
        optim.zero_grad()
        loss = F.mse_loss(preds, y)
        loss.backward()
        optim.step()

    X, y = dataset.X, dataset.y
    preds = network(X)
    loss = F.mse_loss(preds, y)
    print(loss)
