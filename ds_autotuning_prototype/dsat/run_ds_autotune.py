import argparse
import logging
import pathlib
import shutil

import determined as det
from determined.experimental.client import create_experiment
from dsat import constants, utils


def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workspace_name", type=str, default="")
    parser.add_argument("-p", "--project_name", type=str, default="")
    parser.add_argument("-e", "--exp_name", type=str, default="")
    parser.add_argument("-c", "--config_path", type=str, default="")

    parsed_args = parser.parse_args()

    return parsed_args


def main(core_context: det.core.Context, args: argparse.Namespace) -> None:
    is_chief = core_context.distributed.get_rank() == 0
    if is_chief:
        # Launch the multi-Trial DS AT search
        # profiling_results_config holds the model profiling results
        # The path to the original config is passed as an arg to dsat_searcher, as this
        # information is also needed.
        src = pathlib.Path(constants.OUTPUT_FILE_PATH)
        ds_profiler_results = utils.DSProfilerResults(path=src)
        # TODO: Not currently passing the original searcher metric into this experiment.
        profiling_results_config = ds_profiler_results.get_config(
            workspace_name=args.workspace_name,
            project_name=args.project_name,
            exp_name=args.exp_name,
            entrypoint=f"python3 -m dsat.dsat_searcher -c {args.config_path}",
        )
        create_experiment(config=profiling_results_config, model_dir=".")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    distributed = det.core.DistributedContext.from_torch_distributed()
    with det.core.init(distributed=distributed) as core_context:
        args = get_parsed_args()
        main(core_context, args=args)
