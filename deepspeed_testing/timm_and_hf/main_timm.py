import argparse
import logging
from typing import Any, Dict

import attrdict
import data
import deepspeed
import determined as det

import trainer
import models


def parse_args():
    parser = argparse.ArgumentParser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--find_max_batch_size", action="store_true")
    parser.add_argument("--autotuning", action="store_true")
    # Need to absorb (and do nothing with) a local_rank arg when running autotuning.
    parser.add_argument("--local_rank", type=int, default=None)

    args = parser.parse_args()

    return args


def main(core_context, hparams: Dict[str, Any], latest_checkpoint: str) -> None:
    hparams = attrdict.AttrDict(hparams)
    model = models.build_timm_model(
        model_name=hparams.model_name, checkpoint_path_prefix=hparams.checkpoint_path_prefix
    )
    transforms = data.build_timm_transforms(model_name=hparams.model_name)
    args = parse_args()
    kwargs = {
        "core_context": core_context,
        "latest_checkpoint": latest_checkpoint,
        "args": args,
        "model": model,
        "transforms": transforms,
        "dataset_name": hparams.dataset_name,
        "sanity_check": hparams.sanity_check,
    }
    if args.find_max_batch_size:
        trainer.DeepSpeedTrainer.find_max_batch_size(**kwargs)
    elif args.autotuning:
        ds_trainer = trainer.DeepSpeedTrainer(**kwargs)
        ds_trainer.autotuning()
    else:
        ds_trainer = trainer.DeepSpeedTrainer(**kwargs)
        ds_trainer.train_on_cluster()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    latest_checkpoint = info.latest_checkpoint
    hparams = info.trial.hparams
    distributed = det.core.DistributedContext.from_deepspeed()
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, hparams, latest_checkpoint)
