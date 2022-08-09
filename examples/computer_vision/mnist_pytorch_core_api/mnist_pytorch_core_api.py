import logging

import determined as det
import torch
import torch.nn as nn
from attrdict import AttrDict

import models
from trainer import Trainer


def main(core_context, info, model_class: nn.Module = models.MNISTModel) -> None:
    hparams = AttrDict(info.trial.hparams)
    model = model_class(**hparams.model)
    optimizer = torch.optim.Adam(model.parameters(), **hparams.optimizer)
    trainer = Trainer(core_context, info, model, optimizer, **hparams.trainer)
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
