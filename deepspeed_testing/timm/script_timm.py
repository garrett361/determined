import argparse
import contextlib
import json
import os
import sys

from determined.experimental import client

import timm_models
import workspaces


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


parser = argparse.ArgumentParser(description="ImageNet DeepSpeedTrainer Loops")
parser.add_argument("-m", "--master", type=str, default="localhost:8080")
parser.add_argument("-u", "--user", type=str, default="determined")
parser.add_argument("-p", "--password", type=str, default="")
parser.add_argument("-spt", "--slots_per_trial", type=int, default=2)
parser.add_argument("-d", "--dataset_name", type=str, default="imagenette2-160")
parser.add_argument("-mc", "--model_criteria", type=str, default="small")
parser.add_argument("-en", "--experiment_name", type=str, default="")
parser.add_argument("-pn", "--project_name", type=str, default="")
parser.add_argument("-w", "--workspace", type=str, default="DeepSpeed")
parser.add_argument("-mn", "--model_name", type=str, default="")
parser.add_argument(
    "-cpp", "--checkpoint_path_prefix", type=str, default="shared_fs/ensembles/state_dicts/"
)
parser.add_argument("-tmbs", "--train_micro_batch_size_per_gpu", type=int, default=128)
parser.add_argument("-gas", "--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("-lr", "--lr", type=float, default=0.001)
parser.add_argument("-e", "--epochs", type=int, default=1)
parser.add_argument("-sc", "--sanity_check", action="store_true")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("-zs", "--zero_stage", type=int, nargs="+", default=[0])
parser.add_argument("-fl", "--flops_profiler", action="store_true")
parser.add_argument("-a", "--autotuning", type=str, default="")
parser.add_argument("-am", "--autotuning_metric", type=str, default="throughput")
parser.add_argument("-af", "--autotuning_fast", action="store_true")
parser.add_argument("--fp16", action="store_true")
args = parser.parse_args()

# Various sanity checks, hacks, and dynamic fixes.
if len(args.zero_stage) == 1:
    args.zero_stage = args.zero_stage[0]


if args.autotuning:
    assert args.autotuning in (
        "tune",
        "run",
    ), 'autotuning must be either "tune" or "run", when provided'
    assert args.autotuning_metric in (
        "latency",
        "throughput",
        "FLOPS",
    ), 'autotuning_metric must be either "latency", "throughput", or "FLOPS"'

if not args.model_name:
    args.model_name = timm_models.get_model_names_from_criteria(model_criteria=args.model_criteria)[
        0
    ]

# Generate experiment names from command line arguments, if none provided.
generate_exp_names = args.experiment_name == ""
# Append '_test' to the given workspace name, if --test is set.
workspace_name = args.workspace + ("_test" if args.test else "")
# If a non-blank project_name is provided, use that project; otherwise use the dataset_name
project_name = args.project_name or args.dataset_name

if generate_exp_names:
    args.experiment_name = f"{args.model_name}"


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


base_hps = {
    "dataset_name": args.dataset_name,
    "model_criteria": args.model_criteria,
    "model_name": args.model_name,
    "sanity_check": args.sanity_check,
    "checkpoint_path_prefix": args.checkpoint_path_prefix,
}

# The ds_config requires a hack for the `type` key to avoid a conflict with the expected values for
# a `type` field by the searcher. We just capitalize and lower before processing.
ds_config = {
    "train_micro_batch_size_per_gpu": args.train_micro_batch_size_per_gpu,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "steps_per_print": 10,
    "optimizer": {
        "Type": "Adam",
        "params": {"lr": args.lr, "betas": [0.8, 0.999], "eps": 1e-8, "weight_decay": 3e-7},
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

if args.flops_profiler:
    ds_config["flops_profiler"] = flops_profiler

# Autotuning requires various hacks.

autotuning = {
    "enabled": True,
    "metric": args.autotuning_metric,
    "fast": args.autotuning_fast,
}

if args.autotuning:
    ds_config["autotuning"] = autotuning

entrypoint = "python3 deepspeed_single_node_launcher.py"
if args.autotuning:
    entrypoint += f" --autotuning={args.autotuning} --"
    with open("ds_config.json", "w") as f:
        json.dump(ds_config, f)
entrypoint += " python3 main_timm.py"
if args.autotuning:
    entrypoint += " --deepspeed ds_config.json"

config = {
    "entrypoint": entrypoint,
    "name": args.experiment_name,
    "workspace": workspace_name,
    "project": project_name,
    "max_restarts": 0,
    "reproducibility": {"experiment_seed": 42},
    "resources": {"slots_per_trial": args.slots_per_trial},
    "searcher": {
        "name": "single",
        "max_length": args.epochs,
        "metric": "train_top1_acc",
        "smaller_is_better": False,
    },
    "environment": {
        "environment_variables": ["OMP_NUM_THREADS=1"],
        "image": {
            "gpu": "determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-deepspeed-0.7.0-gpu-0.19.4"
        },
    },
    "hyperparameters": {**base_hps, **{"ds_config": ds_config}},
}


with suppress_stdout():
    client.create_experiment(config=config, model_dir=".")
