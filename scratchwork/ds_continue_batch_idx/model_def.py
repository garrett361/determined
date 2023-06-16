import datetime
import os
import time
from typing import Any, Dict

import deepspeed
import filelock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from attrdict import AttrDict
from torch.utils.data import Dataset

from determined.pytorch import DataLoader
from determined.pytorch.deepspeed import (
    DeepSpeedTrial,
    DeepSpeedTrialContext,
    overwrite_deepspeed_config,
)


class BatchIdxDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1_000_000

    def __getitem__(self, index):
        return torch.tensor(index)


class BatchIdxTrial(DeepSpeedTrial):
    def __init__(self, context: DeepSpeedTrialContext) -> None:
        self.context = context
        self.args = AttrDict(self.context.get_hparams())

        model = nn.Linear(10, 10)
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        model_engine, optimizer, __, __ = deepspeed.initialize(
            model=model, model_parameters=parameters, config=self.args.deepspeed_config
        )

        self.model_engine = self.context.wrap_model_engine(model_engine)

    def train_batch(self, iter_dataloader, epoch_idx, batch_idx) -> Dict[str, torch.Tensor]:
        batch = self.context.to_device(next(iter_dataloader))
        print(batch)
        time.sleep(1)
        return {"accuracy": 0}

    def evaluate_batch(self, iter_dataloader, batch_idx) -> Dict[str, Any]:
        return {"accuracy": 0}

    def build_training_data_loader(self) -> Any:
        return DataLoader(
            BatchIdxDataset(),
            batch_size=self.context.train_micro_batch_size_per_gpu,
            shuffle=False,
        )

    def build_validation_data_loader(self) -> Any:
        return DataLoader(
            BatchIdxDataset(),
            batch_size=self.context.train_micro_batch_size_per_gpu,
            shuffle=False,
        )
