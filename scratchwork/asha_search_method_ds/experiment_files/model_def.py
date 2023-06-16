"""
This example shows how to interact with the Determined PyTorch interface to
build a basic MNIST network.

In the `__init__` method, the model and optimizer are wrapped with `wrap_model`
and `wrap_optimizer`. This model is single-input and single-output.

The methods `train_batch` and `evaluate_batch` define the forward pass
for training and evaluation respectively.
"""

from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union, cast

import data
import deepspeed
import torch
from attrdict import AttrDict
from layers import Flatten
from torch import nn

from determined.pytorch import DataLoader
from determined.pytorch.deepspeed import (
    DeepSpeedTrial,
    DeepSpeedTrialContext,
    overwrite_deepspeed_config,
)

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class MNistTrial(DeepSpeedTrial):
    def __init__(self, context: DeepSpeedTrialContext) -> None:
        self.context = context
        self.args = AttrDict(self.context.get_hparams())

        # Create a unique download directory for each rank so they don't overwrite each
        # other when doing distributed training.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.data_downloaded = False

        model = nn.Sequential(
            nn.Conv2d(1, self.context.get_hparam("n_filters1"), 3, 1),
            nn.ReLU(),
            nn.Conv2d(
                self.context.get_hparam("n_filters1"),
                self.context.get_hparam("n_filters2"),
                3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.context.get_hparam("dropout1")),
            Flatten(),
            nn.Linear(144 * self.context.get_hparam("n_filters2"), 128),
            nn.ReLU(),
            nn.Dropout2d(self.context.get_hparam("dropout2")),
            nn.Linear(128, 10),
            nn.LogSoftmax(),
        )
        ds_config = overwrite_deepspeed_config(
            self.args.deepspeed_config, self.args.get("overwrite_deepspeed_args", {})
        )
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_engine, __, __, __ = deepspeed.initialize(
            model=model, model_parameters=parameters, config=ds_config
        )
        self.fp16 = model_engine.fp16_enabled()
        self.model_engine = self.context.wrap_model_engine(model_engine)

    def build_training_data_loader(self) -> DataLoader:
        if not self.data_downloaded:
            self.download_directory = data.download_dataset(
                download_directory=self.download_directory,
                data_config=self.context.get_data_config(),
            )
            self.data_downloaded = True

        train_data = data.get_dataset(self.download_directory, train=True)
        return DataLoader(train_data, batch_size=self.context.train_micro_batch_size_per_gpu)

    def build_validation_data_loader(self) -> DataLoader:
        if not self.data_downloaded:
            self.download_directory = data.download_dataset(
                download_directory=self.download_directory,
                data_config=self.context.get_data_config(),
            )
            self.data_downloaded = True

        validation_data = data.get_dataset(self.download_directory, train=False)
        return DataLoader(validation_data, batch_size=self.context.train_micro_batch_size_per_gpu)

    def train_batch(
        self,
        dataloader_iter: Optional[Iterator[TorchData]],
        epoch_idx: int,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        batch = self.context.to_device(next(dataloader_iter))
        data, labels = batch
        if self.fp16:
            data = data.half()

        output = self.model_engine(data)
        loss = torch.nn.functional.nll_loss(output, labels)

        self.model_engine.backward(loss)
        self.model_engine.step()

        return {"loss": loss}

    def evaluate_batch(
        self, dataloader_iter: Optional[Iterator[TorchData]], batch_idx: int
    ) -> Dict[str, Any]:
        batch = self.context.to_device(next(dataloader_iter))
        data, labels = batch
        if self.fp16:
            data = data.half()

        output = self.model_engine(data)
        validation_loss = torch.nn.functional.nll_loss(output, labels).item()

        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(data)

        return {"validation_loss": validation_loss, "accuracy": accuracy}
