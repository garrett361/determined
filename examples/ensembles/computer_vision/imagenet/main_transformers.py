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
    model_names = timm_models.get_timm_ensembles_of_model_names(
        model_criteria="small", num_base_models=2, num_ensembles=1
    )
    models = timm_models.build_timm_models(
        model_names[0], checkpoint_path_prefix="shared_fs/state_dicts/"
    )
    transforms = data.build_timm_transforms(models=models)

    model = ensemble_transformer.ModelEnsembleTransformer(models=models)
    optimizer = torch.optim.Adam(model.parameters(), **hparams.optimizer)
    train_dataset = data.get_dataset(split="train", name="imagenette2-160", transforms=transforms)
    val_dataset = data.get_dataset(split="val", name="imagenette2-160", transforms=transforms)

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
