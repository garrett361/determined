"""
Generic code for setting up a PyTorchTrial from an `nn.Module` subclass which provides a forward method; expected to
be in model.Model.
"""

import attrdict
from determined.pytorch import (
    DataLoader,
    PyTorchTrial,
    PyTorchTrialContext,
)
from timm.optim import create_optimizer
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import data
import model


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class ModelTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        self.hparams = attrdict.AttrDict(self.context.get_hparams())
        self.model_config = self.hparams.model
        self.optimizer_config = self.hparams.optimizer
        self.transform_config = self.hparams.transform
        self.data_config = attrdict.AttrDict(self.context.get_data_config())
        self.dataset_metadata = data.DATASET_METADATA_BY_NAME[
            self.data_config.dataset_name
        ]

        self.model = self.context.wrap_model(
            model.Model(
                self.dataset_metadata.to_dict(),
                **self.model_config,
            )
        )

        # Use timm's create_xxx factories for the optimizer and scheduler.
        optimizer = create_optimizer(self.optimizer_config, self.model)
        self.optimizer = self.context.wrap_optimizer(optimizer)

    def build_training_data_loader(self) -> DataLoader:
        training_data_loader = self._get_data_loader(train=True)
        return training_data_loader

    def build_validation_data_loader(self) -> DataLoader:
        validation_data_loader = self._get_data_loader(train=False)
        return validation_data_loader

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return {"loss": 0.0}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        return {"validation_loss": 0.0, "accuracy": 0.0}

    def _get_data_loader(self, train: bool) -> DataLoader:
        dataset = FakeDataset(self.context.get_per_slot_batch_size())
        return DataLoader(
            dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            persistent_workers=self.data_config.persistent_workers,
            shuffle=train,
            drop_last=train,
        )


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(0.0)
