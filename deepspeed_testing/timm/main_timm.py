import logging
from typing import Any, Dict

import attrdict
import data
import determined as det
import trainer
import timm_models


def lower_dict_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    lower_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            lower_d[k.lower()] = lower_dict_keys(v)
        else:
            lower_d[k.lower()] = v
    return lower_d


def main(core_context, hparams: Dict[str, Any], latest_checkpoint: str) -> None:
    hparams = attrdict.AttrDict(hparams)
    model = timm_models.build_timm_model(
        model_name=hparams.model_name, checkpoint_path_prefix=hparams.checkpoint_path_prefix
    )
    transforms = data.build_timm_transforms(model_name=hparams.model_name)
    ds_trainer = trainer.DeepSpeedTrainer(
        core_context,
        latest_checkpoint=latest_checkpoint,
        model=model,
        transforms=transforms,
        dataset_name=hparams.dataset_name,
        ds_config=lower_dict_keys(hparams.ds_config),
        sanity_check=hparams.sanity_check,
    )
    ds_trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    latest_checkpoint = info.latest_checkpoint
    hparams = info.trial.hparams
    distributed = det.core.DistributedContext.from_deepspeed()
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, hparams, latest_checkpoint)
