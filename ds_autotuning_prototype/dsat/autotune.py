import argparse
import collections
import copy
import os
from typing import Any, Dict

from determined.experimental import client
from dsat import constants, utils


def parse_args():

    parser = argparse.ArgumentParser(description="DS Autotuning")
    parser.add_argument("-m", "--master", type=str, default="")
    parser.add_argument("-u", "--user", type=str, default="determined")
    parser.add_argument("-p", "--password", type=str, default="")

    parser.add_argument("config_path")
    parser.add_argument("model_dir")
    args = parser.parse_args()
    return args


def get_mem_per_gpu(num_params, total_gpus, fp16_enabled, mp_size, zero_stage):

    # assume the model uses Adam optimizer (GG: inherited assump from DS)
    params_mem = num_params * (2 if fp16_enabled else 4)
    gradients_mem = num_params * (2 if fp16_enabled else 4)
    optimizer_mem = num_params * (16 if fp16_enabled else 8)

    if zero_stage >= 0:
        optimizer_mem = optimizer_mem / total_gpus

    if zero_stage >= 1:
        gradients_mem = gradients_mem / total_gpus

    if zero_stage >= 2:
        params_mem = params_mem / total_gpus

    mem_per_gpu = (params_mem + gradients_mem + optimizer_mem) / mp_size()
    return mem_per_gpu


def run_autotuning(args: argparse.Namespace, config_dict: Dict[str, Any]):
    model_info_profiling_config = copy.deepcopy(config_dict)
    utils.replace_dict(
        model_info_profiling_config["hyperparameters"]["ds_config"],
        constants.MODEL_INFO_PROFILING_DS_CONFIG,
    )

    model_info_profiling_config["searcher"] = constants.SINGLE_SEARCHER_CONFIG
    model_info_profiling_config["name"] += " (model info profile run)"

    project_name = config_dict.get("project", "")
    workspace_name = config_dict.get("workspace", "")
    exp_name = config_dict.get("name", "")
    # Append the autotuning launcher after the original entrypoint.
    # Need distributed launching here to ensure that only the chief launches the follow
    # on script.
    # TODO: Error handling if profiling run fails.
    # NOTE: Currently these additional entrypoints only run non-trivial code on the
    # chief, which is why the torch_distributed launcher is needed.
    model_info_profiling_config["entrypoint"] += (
        "; python3 -m determined.launch.torch_distributed python3 -m dsat.checkpoint_profiling_results"
        "; python3 -m determined.launch.torch_distributed python3 -m dsat.run_ds_autotune"
        f" -p {project_name} -e {exp_name} -w {workspace_name} -c {args.config_path}"
    )
    # TODO: Need to account for case where config isn't in model_dir, in which case
    # we need to pass its path to the `includes` arg of `create_experiment`
    client.create_experiment(config=model_info_profiling_config, model_dir=args.model_dir)


def run_other_experiment(args: argparse.Namespace, config_dict: Dict[str, Any]):
    client.create_experiment(config=config_dict, model_dir=args.model_dir)


def run():
    args = parse_args()

    # Convert config to python dict
    config_dict = utils.get_config_dict_from_yaml_path(args.config_path)

    if not args.master:
        args.master = os.getenv("DET_MASTER", "localhost:8000")

    client.login(master=args.master, user=args.user, password=args.password)

    if config_dict["searcher"]["name"] == "custom":
        run_autotuning(args, config_dict)
    else:
        run_other_experiment(args, config_dict)


if __name__ == "__main__":
    run()
