import logging
from typing import Any, Dict

import determined as det

import attrdict
import data
import ensemble_trainer
import timm_models


def main(core_context, hparams: Dict[str, Any]) -> None:
    hparams = attrdict.AttrDict(hparams)

    model_list = timm_models.build_timm_model_list(
        hparams.model_names, hparams.checkpoint_path_prefix
    )
    trainer = ensemble_trainer.EnsembleTrainer(
        core_context,
        model_list=model_list,
        train_batch_size=hparams.train_batch_size,
        val_batch_size=hparams.val_batch_size,
        dataset_name=hparams.dataset_name,
        ensemble_strategy=hparams.ensemble_strategy,
        ensemble_args=None,
        extra_val_log_metrics=hparams,
        sanity_check=hparams.sanity_check,
        num_combinations=hparams.num_combinations,
        lr=hparams.lr,
        epochs=hparams.epochs,
    )
    print("Building ensemble...")
    trainer.build_ensemble()
    print("Validating ensemble...")
    trainer.validate_ensemble()


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
