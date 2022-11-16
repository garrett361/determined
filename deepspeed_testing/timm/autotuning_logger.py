import argparse
import pathlib
import shutil
import sys

import determined as det


def parse_args():
    parser = argparse.ArgumentParser()
    # Absorb a possible `local_rank` arg from the launcher.
    parser.add_argument("--last_exit_code", type=int)
    args = parser.parse_args()

    return args


def main(core_context: det.core.Context):
    is_chief = core_context.distributed.get_rank() == 0
    if is_chief:
        checkpoint_metadata_dict = {"steps_completed": 0}
        with core_context.checkpoint.store_path(checkpoint_metadata_dict) as (path, storage_id):
            for dir in ("autotuning_exps", "autotuning_results"):
                src_path = pathlib.Path(dir)
                shutil.copytree(
                    src=src_path,
                    dst=pathlib.Path(path).joinpath(dir),
                )


if __name__ == "__main__":
    distributed = det.core.DistributedContext.from_torch_distributed()
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
        args = parse_args()
        sys.exit(args.last_exit_code)
