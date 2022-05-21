from typing import Any, Dict, List, Sequence, Union

import attrdict
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler

from determined.pytorch import (
    DataLoader,
    PyTorchTrial,
    PyTorchTrialContext,
    LRScheduler,
)


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class SimpleDataset(Dataset):
    def __init__(self, size: int = 3) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: Union[int, List[int]]) -> torch.Tensor:
        if isinstance(idx, int):
            idx = [idx]
        return torch.zeros(len(idx))


class CountingLRScheduler(_LRScheduler):
    def __init__(self, optimizer, rank):
        self.steps = 0
        self.rank = rank
        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def step(self) -> None:
        self.steps += 1
        print(f"{self.steps} total steps taken by rank {self.rank} worker")


class StepEveryEpochTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.rank = self.context.distributed.rank
        self.hparams = attrdict.AttrDict(self.context.get_hparams())
        self.model = self.context.wrap_model(nn.Sequential(nn.Linear(1, 1)))

        self.optimizer = self.context.wrap_optimizer(
            torch.optim.RMSprop(self.model.parameters(), lr=self.hparams.lr)
        )

        lr_sch = CountingLRScheduler(self.optimizer, self.rank)
        self.lr_sch = self.context.wrap_lr_scheduler(
            lr_sch, step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH
        )

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        self.context.step_optimizer(self.optimizer)
        return {"loss": torch.zeros(1)}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        return {"val_loss": torch.zeros(1)}

    def build_training_data_loader(self) -> Any:
        train_dataset = SimpleDataset(self.hparams.dataset_size)
        return DataLoader(
            train_dataset, batch_size=self.context.get_per_slot_batch_size()
        )

    def build_validation_data_loader(self) -> Any:
        val_dataset = SimpleDataset(self.hparams.dataset_size)
        return DataLoader(
            val_dataset, batch_size=self.context.get_per_slot_batch_size()
        )
