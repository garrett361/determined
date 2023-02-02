import logging
from typing import Any, Dict, Iterable, List, Optional, Union

import determined as det
import torch
import torch.nn as nn
import torch.nn.functional as F
import train
from attrdict import AttrDict
from mup import MuAdam, MuReadout, MuSGD, make_base_shapes, set_base_shapes
from torch.utils.data import DataLoader, Dataset


class RandIdentityDataset(Dataset):
    """Spits out random identical input/target pairs."""

    def __init__(self, num_records: int, input_dim: int) -> None:
        self.num_records = num_records
        self.input_dim = input_dim

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = torch.randn(self.input_dim)
        return sample, sample


class MinimalModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: Union[int, List[int]], layers: Optional[int] = None
    ) -> None:
        """Simple dimension preserving model."""
        super().__init__()
        self.input_dim = input_dim
        if isinstance(hidden_dims, int):
            assert layers is not None, "layers must be specified if hidden_dims is an int"
            hidden_dims = [hidden_dims for _ in range(layers)]

        hidden_layers = [
            nn.Linear(w_in, w_out)
            for w_in, w_out in zip([self.input_dim] + hidden_dims[:-1], hidden_dims)
        ]
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.readout_layer = MuReadout(hidden_dims[-1], self.input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.hidden_layers:
            outputs = layer(outputs)
            outputs = outputs.relu()
        outputs = self.readout_layer(outputs)
        return outputs


def main(core_context, hparams: AttrDict) -> None:
    input_dim, hidden_dims, layers = (
        hparams.model.input_dim,
        hparams.model.hidden_dims,
        hparams.model.layers,
    )
    base_model = MinimalModel(input_dim=input_dim, hidden_dims=1, layers=layers)
    delta_model = MinimalModel(input_dim=input_dim, hidden_dims=2, layers=layers)
    model = MinimalModel(**hparams.model)
    set_base_shapes(model, base_model, delta=delta_model)
    optimizer = MuAdam(model.parameters(), **hparams.optimizer)
    dataset = RandIdentityDataset(**hparams.dataset)
    criterion = nn.MSELoss()
    trainer = train.Trainer(
        core_context=core_context,
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
    hparams = AttrDict(info.trial.hparams)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, hparams)
