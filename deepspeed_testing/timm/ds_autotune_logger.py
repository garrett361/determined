import argparse
import pathlib
import shutil
import sys

import determined as det
from determined.experimental.client import create_experiment

import utils


def get_parsed_args():
    parser = argparse.ArgumentParser()
    # Absorb a possible `local_rank` arg from the launcher.
    parser.add_argument("--last_exit_code", type=int)
    parser.add_argument("-w", "--workspace_name", type=str)
    parser.add_argument("-p", "--project_name", type=str)
    parser.add_argument("-e", "--exp_name", type=str)
    parser.add_argument("-m", "--model_name", type=str)

    parsed_args = parser.parse_args()

    return parsed_args


def main(core_context: det.core.Context, args: argparse.Namespace) -> None:
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
        ds_autotuning_results = utils.DSAutotuningResults(base_path=pathlib.Path("."))
        grid_search_config = ds_autotuning_results.get_grid_search_config(
            workspace_name=args.workspace_name,
            project_name=args.project_name,
            exp_name=args.exp_name,
            model_name=args.model_name,
            entrypoint="python3 single_ds_logger_exp.py",
        )
        create_experiment(config=grid_search_config, model_dir=".")


if __name__ == "__main__":
    distributed = det.core.DistributedContext.from_torch_distributed()
    with det.core.init(distributed=distributed) as core_context:
        args = get_parsed_args()
        main(core_context, args=args)
        sys.exit(args.last_exit_code)
