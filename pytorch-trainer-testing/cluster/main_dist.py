import logging
from typing import Any, Dict

import determined as det
import torch
import torch.nn as nn
from attrdict import AttrDict
from determined.pytorch import DataLoader
from torch.utils.data import Dataset


class RandDataset(Dataset):
    def __init__(self, num_records: int, dim: int) -> None:
        self.num_records = num_records
        self.records = torch.randn(num_records, dim)

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.records[idx]


class IdentityTrial(det.pytorch.PyTorchTrial):
    """Minimal Trial which attempts to learn the identity transformation."""

    def __init__(self, context: det.pytorch.PyTorchTrialContext) -> None:
        self.context = context

        self.hps = AttrDict(self.context.get_hparams())

        self.model = self.context.wrap_model(nn.Linear(self.hps.dim, self.hps.dim))
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.SGD(self.model.parameters(), lr=self.hps.lr)
        )
        self.criterion = nn.MSELoss()

        self.train_set = RandDataset(self.hps.num_train_records, self.hps.dim)
        self.val_set = RandDataset(self.hps.num_val_records, self.hps.dim)

    def build_training_data_loader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
        )

    def build_validation_data_loader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
        )

    def train_batch(
        self, batch: det.pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        pred = self.model(batch)
        loss = self.criterion(pred, batch)
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        return {"train_loss": loss.item()}

    def evaluate_batch(self, batch: det.pytorch.TorchData, batch_idx: int) -> Dict[str, Any]:
        pred = self.model(batch)
        loss = self.criterion(pred, batch)
        return {"val_loss": loss.item()}


def main() -> None:

    with det.pytorch.init() as train_context:
        trial = IdentityTrial(train_context)
        trainer = det.pytorch.Trainer(trial, train_context)
        trainer.fit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    main()
