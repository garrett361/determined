import logging
from typing import Any, Dict, Iterable, Optional, Union

import determined as det
import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from mup import MuAdam, MuReadout, MuSGD, make_base_shapes, set_base_shapes
from torch.utils.data import DataLoader, Dataset

from ..trainer import Trainer


class RandDataset(Dataset):
    def __init__(self, num_records: int, dim: int) -> None:
        self.num_records = num_records
        self.dim = dim

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randn(self.dim)


class MinimalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int]) -> None:
        """Simple dimension preserving model."""
        super().__init__()
        self.input_dim = input_dim
        hidden_layers = [
            nn.Linear(w_in, w_out)
            for w_in, w_out in zip([self.input_dim] + hidden_layers[:-1], hidden_layers)
        ]
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.readout_layer = MuReadout(hidden_layers[-1], self.input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.hidden_layers:
            outputs = layer(outputs)
            outputs = outputs.relu()
        outputs = self.readout_layer(outputs)
        return outputs


def main(core_context, info) -> None:
    hparams = AttrDict(info.trial.hparams)
    model = MinimalModel(**hparams.model)
    optimizer = MuAdam(model.parameters(), **hparams.optimizer)
    dataset = RandDataset(**hparams.dataset)
    criterion = nn.MSELoss()
    trainer = Trainer(
        core_context=core_context,
        info=info,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        criterion=criterion,
        **hparams.trainer
    )
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, info)
