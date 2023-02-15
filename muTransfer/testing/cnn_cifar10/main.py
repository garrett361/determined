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
from torchvision import datasets, transforms

DATA_ROOT = "/run/determined/workdir/shared_fs/data"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root=DATA_ROOT, train=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def main(core_context, hparams: AttrDict, latest_checkpoint: Optional[str] = None) -> None:
    seed_everything(seed=hparams.random_seed)
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
    dataset = mnist_train
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
