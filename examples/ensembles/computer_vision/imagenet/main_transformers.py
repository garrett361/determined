import logging

import determined as det
import torch
from attrdict import AttrDict

import data
import ensemble_transformer
import timm_models
from trainer import Trainer


def main(core_context, info) -> None:
    hparams = AttrDict(info.trial.hparams)

    model_class = ensemble_transformer.TimmModelEnsembleTransformer
    optimizer_class = torch.optim.SGD

    transforms = data.build_timm_transforms(model_names=hparams.model.model_names)
    train_dataset = data.get_dataset(
        split="train", dataset_name=hparams.data.dataset_name, transforms=transforms
    )
    val_dataset = data.get_dataset(
        split="val", dataset_name=hparams.data.dataset_name, transforms=transforms
    )

    trainer = Trainer(
        core_context, info, model_class, optimizer_class, train_dataset, val_dataset, hparams
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
