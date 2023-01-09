import argparse
import logging
import pathlib
import shutil

import determined as det
from determined.experimental.client import create_experiment
from dsat import constants, utils


def main(core_context: det.core.Context) -> None:
    is_chief = core_context.distributed.get_rank() == 0
    if is_chief:
        # Save the profile results as a checkpoint of the calling Trial (Ryan wouldn't approve).
        checkpoint_metadata_dict = {"steps_completed": 0}
        with core_context.checkpoint.store_path(checkpoint_metadata_dict) as (
            path,
            _,
        ):
            src = pathlib.Path(constants.OUTPUT_FILE_PATH)
            dst = pathlib.Path(path).joinpath(src.name)
            shutil.copy(src=src, dst=dst)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    distributed = det.core.DistributedContext.from_torch_distributed()
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
