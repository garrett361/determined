import argparse
import contextlib
import json
import itertools
import os
import sys

from determined.experimental import client
import pandas as pd

import models
import workspaces
import utils

from constants import (
    AUTOTUNINGS,
    AUTOTUNING_END_PROFILE_STEP,
    AUTOTUNING_START_PROFILE_STEP,
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    DS_CONFIG_PATH,
    FLOPS_PROFILER_OUTPUT_PATH,
    MAX_STEPS,
    METRICS,
    FLOPS_PROFILE_STEP,
    TASKS,
    TUNER_TYPES,
)


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

    parser = argparse.ArgumentParser(description="HF GLUE DeepSpeedTrainer Loops")
    parser.add_argument("-m", "--master", type=str, default="localhost:8080")
    parser.add_argument("-u", "--user", type=str, default="determined")
    parser.add_argument("-p", "--password", type=str, default="")
    parser.add_argument("-slots", "--slots_per_trial", type=int, nargs="+", default=[2])
    parser.add_argument("-dn", "--dataset_name", type=str)
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-w", "--workspace", type=str, default="DeepSpeed")
    parser.add_argument("-mn", "--model_name", type=str, nargs="+")
    parser.add_argument("-ens", "--exp_name_suffix", type=str, nargs="+", default=[""])
    parser.add_argument(
        "-cpp", "--checkpoint_path_prefix", type=str, default="shared_fs/ensembles/state_dicts/"
    )
    parser.add_argument("-ad", "--allow_duplicates", action="store_true")

    parser.add_argument("-tmbs", "--train_micro_batch_size_per_gpu", type=int, default=128)
    parser.add_argument("-gas", "--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("-lr", "--lr", type=float, default=1e-10)
    parser.add_argument("-e", "--epochs", type=int, default=None)
    parser.add_argument("-s", "--steps", type=int, default=None)
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-zs", "--zero_stage", type=int, nargs="+", default=[0])
    parser.add_argument("-prof", "--flops_profiler", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("-a", "--autotuning", type=str, default="")
    parser.add_argument("-sm", "--searcher_metric", type=str, nargs="+", default=["throughput"])
    parser.add_argument(
        "-att", "--autotuning_tuner_type", type=str, nargs="+", default=["model_based"]
    )
    parser.add_argument("-af", "--autotuning_fast", action="store_true")
    parser.add_argument("-ant", "--autotuning_num_trials", type=int, default=50)
    parser.add_argument("-aes", "--autotuning_tuner_early_stopping", type=int, default=5)

    parser.add_argument("-fmbs", "--find_max_batch_size", action="store_true")
    parser.add_argument("--task", type=str)

    args = parser.parse_args()
    return args


def exp_name_and_config_generator(args):
    """
    Loops over provided autotuning_tuner_type and searcher_metric values and yields
    exp_name, config tuples.
    """
    # Initial sanity checks
    # Various sanity checks, hacks, and dynamic fixes.
    assert args.task in TASKS, f"Task must be one of {TASKS}"
    if not args.autotuning and len(args.zero_stage) == 1:
        args.zero_stage = args.zero_stage[0]
    assert (
        sum(bool(x) for x in (args.autotuning, args.flops_profiler)) <= 1
    ), f"Only one of autotuning, flops_profiler, find_max_batch_size can be set."

    if args.autotuning:
        assert args.autotuning in AUTOTUNINGS, f"autotuning must be one of {AUTOTUNINGS}"

    if not args.dataset_name:
        args.dataset_name = DEFAULT_DATASETS[args.task]
    if not args.model_name:
        args.model_name = DEFAULT_MODELS[args.task]

    assert not (args.epochs and args.steps), "Only one of epochs or steps may be provided."
    if args.epochs is None and args.steps is None:
        args.steps = MAX_STEPS

    # Generate experiment names from command line arguments
    # Append '_test' to the given workspace name, if --test is set.
    workspace_name = args.workspace + ("_test" if args.test else "")

    with suppress_stdout():
        client.login(master=args.master, user=args.user, password=args.password)

    workspace = workspaces.Workspace(
        workspace_name=workspace_name,
        master_url=args.master,
        username=args.user,
        password=args.password,
        create_workspace=True,
    )
    existing_project_names = workspace.get_all_project_names()

    existing_trials_df = (
        pd.DataFrame() if args.allow_duplicates else workspace.get_trial_best_val_results_df()
    )

    # Check if a Trial with the same exp_name already exists.
    existing_exp_names = set()
    if not existing_trials_df.empty:
        for row, vals in existing_trials_df.iterrows():
            existing_exp_names.add(vals.exp_name)

    # Check for "all" in autotuning_tuner_type and searcher_metric
    if args.autotuning_tuner_type == ["all"]:
        args.autotuning_tuner_type = TUNER_TYPES
    if args.searcher_metric == ["all"]:
        args.searcher_metric = METRICS
    if args.model_name == ["all"]:
        args.model_name = models.get_model_names(args.task)

    for model_name, tuner_type, metric, slots_per_trial, suffix in itertools.product(
        args.model_name,
        args.autotuning_tuner_type,
        args.searcher_metric,
        args.slots_per_trial,
        args.exp_name_suffix,
    ):
        # Dynamically generate project and experiment names.
        if args.project_name:
            project_name = args.project_name
        else:
            project_name = model_name
            if args.autotuning:
                project_name += f".autotuning"
            if args.flops_profiler:
                project_name += ".flops_profiler"
            if project_name not in existing_project_names:
                existing_project_names.append(project_name)
                workspace.create_project(project_name)

        exp_name = f"{slots_per_trial}GPU.{model_name}.{args.dataset_name}"
        if args.fp16:
            exp_name += ".fp16"
        if args.autotuning:
            exp_name += f".{metric}.{tuner_type}"
            if args.autotuning_fast:
                exp_name += ".fast"
        if suffix:
            exp_name += f".{suffix}"

        # Various sanity checks, hacks, and dynamic fixes.
        if args.autotuning:
            assert metric in METRICS, f"metric must be one of {METRICS}"
            assert tuner_type in TUNER_TYPES, f"tuner_type must be one of {TUNER_TYPES}"

        if exp_name in existing_exp_names:
            print(f"Skipping {exp_name}: already exists.")
            continue

        # Useful hps to track when analyzing results or which are used downstream.
        base_hps = {
            "dataset_name": args.dataset_name,
            "model_name": model_name,
            "checkpoint_path_prefix": args.checkpoint_path_prefix,
            "slots_per_trial": slots_per_trial,
        }

        # The native ds_config requires `type` key for various entries which conflicts with
        # the special behavior for the `type` key with our searcher.  We write the DS_CONFIG_PATH
        # file with the lower case type and also log the ds_config as part of the HPs with
        # upper-cased TYPE keys for easier Web UI tracking.
        train_micro_batch_size_per_gpu = (
            args.train_micro_batch_size_per_gpu if args.task == "timm" else "auto"
        )
        ds_config = {
            "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": args.lr,
                },
            },
            "zero_optimization": {"stage": args.zero_stage},
        }
        if args.task == "timm":
            ds_config["fp16"] = {
                "enabled": args.fp16,
                "initial_scale_power": 8,  # Trying to prevent loss scale resizing.
            }

        # Series of optional DeepSpeed configs that can be added in.
        flops_profiler = {
            "enabled": True,
            "profile_step": FLOPS_PROFILE_STEP,  # Changed from the default of 1.
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": FLOPS_PROFILER_OUTPUT_PATH,
        }

        autotuning = {
            "enabled": True,
            "metric": metric,
            "fast": args.autotuning_fast,
            "tuner_early_stopping": args.autotuning_tuner_early_stopping,
            "tuner_type": tuner_type,
            "tuner_num_trials": args.autotuning_num_trials,
            "start_profile_step": AUTOTUNING_START_PROFILE_STEP,
            "end_profile_step": AUTOTUNING_END_PROFILE_STEP,
        }

        if args.flops_profiler:
            ds_config["flops_profiler"] = flops_profiler

        if args.autotuning:
            ds_config["autotuning"] = autotuning

        with open(DS_CONFIG_PATH, "w") as f:
            json.dump(ds_config, f)

        # Perform the case hack explained above.
        ds_config = utils.upper_case_dict_key(ds_config, "type")

        entrypoint = "python3 deepspeed_single_node_launcher.py "
        if args.autotuning:
            entrypoint += f"--autotuning {args.autotuning} -- "

        run_glue_cmd_list = [
            "./startup-hook_hf_extras.sh",
            f"shared_fs/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed {DS_CONFIG_PATH}",
            f"--model_name_or_path {model_name}",
            f"--task_name {args.dataset_name}",
            "--do_train",
            "--max_seq_length 128",
            f"--learning_rate {args.lr} ",
            f"--num_train_epochs {args.epochs}",
            f"--output_dir ./output_dir",
            "--save_steps 0",
            "--overwrite_output_dir;",
        ]

        run_clm_cmd_list = [
            "./startup-hook_hf_extras.sh",
            f"shared_fs/transformers/examples/pytorch/language-modeling/run_clm.py --deepspeed {DS_CONFIG_PATH}",
            f"--model_name_or_path {model_name}",
            f"--dataset_name {args.dataset_name}",
            "--dataset_config_name wikitext-2-raw-v1",
            "--do_train",
            "--fp16" if args.fp16 else "",
            f"--learning_rate {args.lr} ",
            f"--num_train_epochs {args.epochs}",
            f"--output_dir ./output_dir",
            "--save_steps 0",
            "--overwrite_output_dir",
            '--save_strategy "no";',
        ]

        run_timm_cmd_list = [
            "main_timm.py",
            f"--deepspeed_config {DS_CONFIG_PATH}",
            f"{'--find_max_batch_size' if args.find_max_batch_size else ''}",
            f"{'--autotuning' if args.autotuning else ''}",
        ]

        cmd_list_dict = {
            "timm": run_timm_cmd_list,
            "hf_glue": run_glue_cmd_list,
            "hf_clm": run_clm_cmd_list,
        }

        entrypoint += " ".join(cmd_list_dict[args.task])

        config = {
            "entrypoint": entrypoint,
            "name": exp_name,
            "workspace": workspace_name,
            "project": project_name,
            "max_restarts": 3,
            "checkpoint_storage": {"save_trial_latest": 1000},  # Only saving text files.
            "reproducibility": {"experiment_seed": 42},
            "resources": {"slots_per_trial": slots_per_trial},
            "searcher": {
                "name": "single",
                "max_length": args.steps or args.epochs,
                "metric": metric,
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
    try:
        print("", 80 * "*", f"Successfully submitted {idx + 1} experiments.", 80 * "~", sep="\n")
    except NameError:
        print("", 80 * "*", "No experiments were submitted.", 80 * "~", sep="\n")
