import argparse
import contextlib
import json
import itertools
import os
import sys

from determined.experimental import client
import pandas as pd

import timm_models
import workspaces
import utils


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def parse_args():

    parser = argparse.ArgumentParser(description="ImageNet DeepSpeedTrainer Loops")
    parser.add_argument("-m", "--master", type=str, default="localhost:8080")
    parser.add_argument("-u", "--user", type=str, default="determined")
    parser.add_argument("-p", "--password", type=str, default="")
    parser.add_argument("-slots", "--slots_per_trial", type=int, nargs="+", default=[2])
    parser.add_argument("-dn", "--dataset_name", type=str, default="mini_imagenet")
    parser.add_argument("-pn", "--project_name", type=str, default="")
    parser.add_argument("-w", "--workspace", type=str, default="DeepSpeed")
    parser.add_argument("-mn", "--model_name", type=str, nargs="+", default=["efficientnet_b0"])
    parser.add_argument(
        "-cpp", "--checkpoint_path_prefix", type=str, default="shared_fs/ensembles/state_dicts/"
    )
    parser.add_argument("-ad", "--allow_duplicates", action="store_true")

    parser.add_argument("-tmbs", "--train_micro_batch_size_per_gpu", type=int, default=128)
    parser.add_argument("-gas", "--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("-lr", "--lr", type=float, default=1e-10)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-sc", "--sanity_check", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-zs", "--zero_stage", type=int, nargs="+", default=[0])
    parser.add_argument("-fl", "--flops_profiler", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("-a", "--autotuning", type=str, default="")
    parser.add_argument("-am", "--autotuning_metric", type=str, nargs="+", default=["throughput"])
    parser.add_argument(
        "-att", "--autotuning_tuner_type", type=str, nargs="+", default=["model_based"]
    )
    parser.add_argument("-af", "--autotuning_fast", action="store_true")
    parser.add_argument("-ant", "--autotuning_num_trials", type=int, default=50)
    parser.add_argument("-aes", "--autotuning_tuner_early_stopping", type=int, default=5)

    args = parser.parse_args()
    return args


def exp_name_and_config_generator(args):
    """
    Loops over provided autotuning_tuner_type and autotuning_metric values and yields
    exp_name, config tuples.
    """
    # Generate experiment names from command line arguments
    # Append '_test' to the given workspace name, if --test is set.
    workspace_name = args.workspace + ("_test" if args.test else "")
    # If a non-blank project_name is provided, use that project; otherwise use the dataset_name
    project_name = args.project_name or args.dataset_name
    with suppress_stdout():
        client.login(master=args.master, user=args.user, password=args.password)

    workspace = workspaces.Workspace(
        workspace_name=workspace_name,
        master_url=args.master,
        username=args.user,
        password=args.password,
        create_workspace=True,
    )
    workspace.create_project(project_name)

    existing_trials_df = (
        pd.DataFrame()
        if args.allow_duplicates
        else workspace.get_all_trials_df(project_names=project_name)
    )

    # Check if a Trial with the same strategy and model names already exists
    existing_exp_names = set()
    if not existing_trials_df.empty:
        for hp_dict in existing_trials_df.hparams:
            try:
                existing_exp_names.add(hp_dict["exp_name"])
            except KeyError:
                continue

    # Check for "all" in autotuning_tuner_type and autotuning_metric
    if args.autotuning_tuner_type == ["all"]:
        args.autotuning_tuner_type = ["gridsearch", "random", "model_based"]
    if args.autotuning_metric == ["all"]:
        args.autotuning_metric = ["throughput", "latenc y", "FLOPS_per_gpu"]
    if args.model_name == ["all"]:
        args.model_name = timm_models.get_model_names()

    for model_name, tuner_type, metric, slots_per_trial in itertools.product(
        args.model_name, args.autotuning_tuner_type, args.autotuning_metric, args.slots_per_trial
    ):
        # Various sanity checks, hacks, and dynamic fixes.
        if not args.autotuning and len(args.zero_stage) == 1:
            args.zero_stage = args.zero_stage[0]

        if args.autotuning:
            assert args.autotuning in (
                "tune",
                "run",
            ), 'autotuning must be either "tune" or "run", when provided'
            assert metric in (
                "latency",
                "throughput",
                "FLOPS_per_gpu",
            ), f'autotuning_metric must be either "latency", "throughput", or "FLOPS_per_gpu", not {metric}'
            assert tuner_type in (
                "gridsearch",
                "random",
                "model_based",
            ), f'autotuning_tuner_type must be either "gridsearch", "random", "model_based", not {tuner_type}'

        # Generate a useful experiment name dynamically
        exp_name = f"{model_name}"
        if args.fp16:
            exp_name += ".fp16"
        if args.autotuning_fast:
            exp_name += ".fast"
        if args.autotuning:
            exp_name += f".{metric}.{tuner_type}"
        exp_name += f".{slots_per_trial}GPU"

        if exp_name in existing_exp_names:
            print(f"Skipping {exp_name}: already exists.")
            continue

        base_hps = {
            "dataset_name": args.dataset_name,
            "model_name": model_name,
            "sanity_check": args.sanity_check,
            "checkpoint_path_prefix": args.checkpoint_path_prefix,
        }

        # The native ds_config requires `type` key for various entries which conflicts with
        # the special behavior for the `type` key with our searcher.  We write the ds_config.json
        # file with the lower case type and also log the ds_config as part of the HPs with upper-cased TYPE
        # keys for easier Web UI tracking.

        ds_config = {
            "train_micro_batch_size_per_gpu": args.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": args.lr,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7,
                },
            },
            "fp16": {
                "enabled": args.fp16,
            },
            "zero_optimization": {"stage": args.zero_stage},
        }

        # Series of optional DeepSpeed configs that can be added in.
        flops_profiler = {
            "enabled": True,
            "profile_step": 10,
            "module_depth": -1,
            "top_modules": 10,
            "detailed": True,
            "output_file": None,
        }

        autotuning = {
            "enabled": True,
            "metric": metric,
            "fast": args.autotuning_fast,
            "tuner_early_stopping": args.autotuning_tuner_early_stopping,
            "tuner_type": tuner_type,
            "tuner_num_trials": args.autotuning_num_trials,
        }

        if args.flops_profiler:
            ds_config["flops_profiler"] = flops_profiler

        if args.autotuning:
            ds_config["autotuning"] = autotuning

        with open("ds_config.json", "w") as f:
            json.dump(ds_config, f)

        # Perform the case hack explained above.
        ds_config = utils.upper_case_dict_key(ds_config, "type")

        entrypoint = "python3 deepspeed_single_node_launcher.py"
        if args.autotuning:
            entrypoint += f" --autotuning {args.autotuning} --"
        entrypoint += " main_timm.py --deepspeed_config ds_config.json;"
        if args.autotuning:
            entrypoint += (
                f"python3 -m determined.launch.torch_distributed python3 "
                f"ds_autotune_logger.py --last_exit_code $? -w {workspace_name} "
                f"-p {project_name} -e {exp_name} -m {model_name}"
            )

        config = {
            "entrypoint": entrypoint,
            "name": exp_name,
            "workspace": workspace_name,
            "project": project_name,
            "max_restarts": 0,
            "reproducibility": {"experiment_seed": 42},
            "resources": {"slots_per_trial": slots_per_trial},
            "searcher": {
                "name": "single",
                "max_length": args.epochs,
                "metric": "metric",
                "smaller_is_better": False,
            },
            "environment": {
                "environment_variables": ["OMP_NUM_THREADS=1"],
                "image": {
                    "gpu": "determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-deepspeed-0.7.0-gpu-0.19.4"
                },
            },
            "hyperparameters": {**base_hps, **{"ds_config": ds_config}, **{"exp_name": exp_name}},
        }
        yield exp_name, config


if __name__ == "__main__":
    args = parse_args()
    for idx, (exp_name, config) in enumerate(exp_name_and_config_generator(args)):
        print(f"Submitting experiment {idx + 1}: {exp_name}")
        with suppress_stdout():
            client.create_experiment(config=config, model_dir=".")
