import logging
import os

from datasets import load_from_disk

import determined as det
from determined.experimental.client import get_checkpoint


def main(core_context) -> None:
    logging.info(f"ENVVARS: {os.environ}")
    ckpt_uuid = "f1937b0d-f56b-4653-99e5-06b4e11425b6"
    try:
        logging.info("Attemping download with get_checkpoint")
        dataset_checkpoint = get_checkpoint(ckpt_uuid)
        dataset_checkpoint_path = dataset_checkpoint.download()
        logging.info(str(dataset_checkpoint_path))
        dset = load_from_disk(dataset_checkpoint_path)
    except Exception as e:
        logging.info(f"Hit exception {e}, attempting to download with restore_path")
        with core_context.checkpoint.restore_path(ckpt_uuid) as path:
            dset = load_from_disk(dataset_checkpoint_path)
            logging.info(f"Path: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_deepspeed()
    except KeyError:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)