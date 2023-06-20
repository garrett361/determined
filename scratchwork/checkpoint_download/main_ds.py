import logging
import os

import determined as det
from determined.experimental.client import get_checkpoint


def main(core_context) -> None:
    print(f"ENVVARS: {os.environ}")
    dataset_checkpoint = get_checkpoint("f1937b0d-f56b-4653-99e5-06b4e11425b6")
    dataset_checkpoint_path = dataset_checkpoint.download()
    print(str(dataset_checkpoint_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_deepspeed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
