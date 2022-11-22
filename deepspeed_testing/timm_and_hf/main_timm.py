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
    # Absorb a possible `local_rank` arg from the launcher.
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local rank passed from distributed launcher"
    )
    parser.add_argument("-fmbs", "--find_max_batch_size", action="store_true")

    args = parser.parse_args()

    return args


def main(core_context, hparams: Dict[str, Any], latest_checkpoint: str) -> None:
    hparams = attrdict.AttrDict(hparams)
    model = models.build_timm_model(
        model_name=hparams.model_name, checkpoint_path_prefix=hparams.checkpoint_path_prefix
    )
    transforms = data.build_timm_transforms(model_name=hparams.model_name)
    args = parse_args()
    print(args)
    if args.find_max_batch_size:
        max_batch_size = trainer.DeepSpeedTrainer.find_max_batch_size(
            core_context=core_context,
            latest_checkpoint=latest_checkpoint,
            args=args,
            model=model,
            transforms=transforms,
            dataset_name=hparams.dataset_name,
            sanity_check=hparams.sanity_check,
        )
        print(f"Max batch size: {max_batch_size}")
    else:
        ds_trainer = trainer.DeepSpeedTrainer(
            core_context=core_context,
            latest_checkpoint=latest_checkpoint,
            args=args,
            model=model,
            transforms=transforms,
            dataset_name=hparams.dataset_name,
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