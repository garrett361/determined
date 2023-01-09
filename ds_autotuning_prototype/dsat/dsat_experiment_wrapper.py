import argparse
import copy
import logging
import pathlib
import shutil
from typing import Any, Dict

import determined as det
from determined.experimental import client
from dsat import constants, utils


def main(core_context: det.core.Context) -> None:
    info = det.get_cluster_info()
    # Hacked things so that the hparams are just the full original config.
    original_config_dict = info.trial.hparams
    # I don't think the below copy is needed, but just being safe.
    profiling_config_dict = copy.deepcopy(original_config_dict)
    for op in core_context.searcher.operations():
        # Submit the actual Trial of interest and wait on its results.
        profiling_config_dict["name"] += " (individual autotuning trial)"
        profiling_config_dict["searcher"] = {
            "name": "single",
            "metric": profiling_config_dict["searcher"]["metric"],
            "max_length": op.length,
            "smaller_is_better": profiling_config_dict["searcher"].get("smaller_is_better", True),
        }
        profiling_config_dict["hyperparameters"]["ds_config"][
            "flops_profiler"
        ] = constants.FLOPS_PROFILER_CONFIG
        profiling_config_dict[
            "entrypoint"
        ] += "; python3 -m determined.launch.torch_distributed python3 -m dsat.checkpoint_profiling_results"
        exp = client.create_experiment(config=profiling_config_dict, model_dir=".")
        exp_exit_status = exp.wait()  # TODO: Error handling.

        # Get the results from the checkpoint (super hacky).
        trial = exp.get_trials()[0]
        ckpt = trial.select_checkpoint(latest=True)
        ckpt.download(path=".")
        profiler_results = utils.DSProfilerResults(path=constants.OUTPUT_FILE_PATH)
        results_dict = profiler_results.get_results_dict()
        print("RESULTS_DICT", results_dict)

        # Report the results to the searcher and Web UI.
        core_context.train.report_validation_metrics(steps_completed=0, metrics=results_dict)
        op.report_completed(results_dict[profiling_config_dict["searcher"]["metric"]])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    with det.core.init(distributed=None) as core_context:
        main(core_context)
