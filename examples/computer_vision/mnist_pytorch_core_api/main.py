import logging

import determined as det
import torch
import torch.nn as nn
from attrdict import AttrDict

import data
import models
from trainer import Trainer


def main(core_context, info) -> None:
    hparams = AttrDict(info.trial.hparams)
    model = models.MNISTModel(**hparams.model)
    optimizer = torch.optim.Adam(model.parameters(), **hparams.optimizer)
    train_dataset = data.get_mnist_dataset(train=True)
    val_dataset = data.get_mnist_dataset(train=False)

    trainer = Trainer(
        core_context, info, model, optimizer, train_dataset, val_dataset, **hparams.trainer
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
