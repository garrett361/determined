import logging
from typing import Any, Dict

import attrdict
import data
import determined as det
import trainer
import timm_models


def main(core_context, hparams: Dict[str, Any]) -> None:
    hparams = attrdict.AttrDict(hparams)
    models = timm_models.build_timm_models(
        model_names=hparams.model_names, checkpoint_path_prefix=hparams.checkpoint_path_prefix
    )
    transforms = data.build_timm_transforms(model_names=hparams.model_names)
    ds_trainer = trainer.DeepSpeedTrainer(
        core_context,
        models=models,
        transforms=transforms,
        train_batch_size=hparams.train_batch_size,
        val_batch_size=hparams.val_batch_size,
        dataset_name=hparams.dataset_name,
        sanity_check=hparams.sanity_check,
        lr=hparams.lr,
        epochs=hparams.epochs,
    )
    ds_trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    hparams = info.trial.hparams
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, hparams)
