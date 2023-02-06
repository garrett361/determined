import logging
import random
from typing import Any, Dict, Iterable, List, Optional, Union

import determined as det
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import train
from attrdict import AttrDict
from mup import MuAdam, MuAdamW, MuReadout, MuSGD, set_base_shapes
from tensorflow._api.v2.train import latest_checkpoint
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Dataset


class RandIdentityDataset(Dataset):
    """Spits out random identical input/target pairs."""

    def __init__(self, num_records: int, input_dim: int) -> None:
        self.records = torch.randn(num_records, self.input_dim)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.records[idx]
        return sample, sample


class MinimalModel(nn.Module):
    def __init__(
        self,
        use_mutransfer: bool,
        input_dim: int,
        width_multiplier: int,
        num_hidden_layers: Optional[int] = None,
    ) -> None:
        """Simple dimension preserving model."""
        super().__init__()
        self.input_dim = input_dim
        hidden_dims = [width_multiplier for _ in range(num_hidden_layers)]

        hidden_layers = [
            nn.Linear(w_in, w_out)
            for w_in, w_out in zip([self.input_dim] + hidden_dims[:-1], hidden_dims)
        ]
        self.hidden_layers = nn.ModuleList(hidden_layers)
        readout_class = MuReadout if use_mutransfer else nn.Linear
        self.readout_layer = readout_class(hidden_dims[-1], self.input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.hidden_layers:
            outputs = layer(outputs)
            outputs = outputs.relu()
        outputs = self.readout_layer(outputs)
        return outputs


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def main(core_context, hparams: AttrDict, latest_checkpoint: Optional[str] = None) -> None:
    seed_everything(seed=hparams.get("radom_seed", 42))
    input_dim, num_hidden_layers = (
        hparams.model.input_dim,
        hparams.model.num_hidden_layers,
    )
    model = MinimalModel(use_mutransfer=hparams.use_mutransfer, **hparams.model)
    if hparams.use_mutransfer:
        logging.info("Using muTransfer")
        base_model = MinimalModel(
            use_mutransfer=hparams.use_mutransfer,
            input_dim=input_dim,
            width_multiplier=1,
            num_hidden_layers=num_hidden_layers,
        )
        delta_model = MinimalModel(
            use_mutransfer=hparams.use_mutransfer,
            input_dim=input_dim,
            width_multiplier=2,
            num_hidden_layers=num_hidden_layers,
        )
        set_base_shapes(model, base_model, delta=delta_model)
    optimizer_class_dict = {
        "sgd": MuSGD if hparams.use_mutransfer else SGD,
        "adam": MuAdam if hparams.use_mutransfer else Adam,
        "adamw": MuAdamW if hparams.use_mutransfer else AdamW,
    }
    optimizer_name = hparams.optimizer_name
    optimizer = optimizer_class_dict[optimizer_name](model.parameters(), **hparams.optimizer)
    dataset = RandIdentityDataset(**hparams.dataset)
    criterion = nn.MSELoss()
    trainer = train.Trainer(
        core_context=core_context,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        criterion=criterion,
        latest_checkpoint=latest_checkpoint,
        **hparams.trainer
    )
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    latest_checkpoint = info.latest_checkpoint
    hparams = AttrDict(info.trial.hparams)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, hparams, latest_checkpoint)
