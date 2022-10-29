import argparse
import contextlib
import math
import os
import sys
from typing import Dict

from determined.experimental import client
import pandas as pd
import re

import strategies
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
parser.add_argument("-d", "--dataset_name", type=str, default="imagenette2-160")
parser.add_argument("-mc", "--model_criteria", type=str, default="small")
parser.add_argument("-en", "--experiment_name", type=str, default="")
parser.add_argument("-pn", "--project_name", type=str, default="")
parser.add_argument("-w", "--workspace", type=str, default="DeepSpeed")
parser.add_argument("-mn", "--model_name", type=str, default="")
parser.add_argument(
    "-cpp", "--checkpoint_path_prefix", type=str, default="shared_fs/ensembles/state_dicts/"
)
parser.add_argument("-tb", "--train_batch_size", type=int, default=256)
parser.add_argument("-vb", "--val_batch_size", type=int, default=256)
parser.add_argument("-lr", "--lr", type=float, default=0.001)
parser.add_argument("-e", "--epochs", type=int, default=1)
parser.add_argument("-sc", "--sanity_check", action="store_true")
parser.add_argument("-ad", "--allow_duplicates", action="store_true")
parser.add_argument("-du", "--delete_unvalidated", action="store_true")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("-nsc", "--no_safety_check", action="store_true")
args = parser.parse_args()

if not args.model_name:
    args.model_name = timm_models.get_model_names_from_criteria(model_criteria=args.model_criteria)[
        0
    ]

# Generate experiment names from command line arguments, if none provided.
generate_names = args.experiment_name == ""
# Append '_test' to the given workspace name, if --test is set.
workspace_name = args.workspace + ("_test" if args.test else "")
# If a non-blank project_name is provided, use that project; otherwise use the dataset_name
project_name = args.project_name or args.dataset_name

with suppress_stdout():
    client.login(master=args.master, user=args.user, password=args.password)


# Safety check for accidentally running a lot of experiments.
# if not args.no_safety_check and num_experiments >= 100:
#     confirm = input(f"Submit {num_experiments} experiments? [yes/N]\n")
#     if confirm != "yes":
#         sys.exit("Cancelling experiment creation.")

workspace = workspaces.Workspace(
    workspace_name=workspace_name,
    master_url=args.master,
    username=args.user,
    password=args.password,
    create_workspace=True,
)
workspace.create_project(project_name)
if args.delete_unvalidated:
    workspace.delete_experiments_with_unvalidated_trials(
        projects_to_delete_from=project_name, safe_mode=False
    )

existing_trials_df = (
    pd.DataFrame()
    if args.allow_duplicates
    else workspace.get_trial_best_val_results_df(project_names=project_name)
)

base_hps = {
    "train_batch_size": args.train_batch_size,
    "val_batch_size": args.val_batch_size,
    "dataset_name": args.dataset_name,
    "model_criteria": args.model_criteria,
    "sanity_check": args.sanity_check,
    "checkpoint_path_prefix": args.checkpoint_path_prefix,
    "lr": None,
    "model_name": args.model_name,
}

ds_config = {
  "train_batch_size": 16,
  "steps_per_print": 2000,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 0.001,
      "warmup_num_steps": 1000
    }
  },
  "gradient_clipping": 1.0,
  "prescale_gradients": False,
  "fp16": {
      "enabled": True,
      "fp16_master_weights_and_grads": False,
      "loss_scale": 0,
      "loss_scale_window": 500,
      "hysteresis": 2,
      "min_loss_scale": 1,
      "initial_scale_power": 15
  },
  "wall_clock_breakdown": False,
  "zero_optimization": {
      "stage": 0,
      "allgather_partitions": True,
      "reduce_scatter": True,
      "allgather_bucket_size": 50000000,
      "reduce_bucket_size": 50000000,
      "overlap_comm": True,
      "contiguous_gradients": True,
      "cpu_offload": False
  }
}

config = {
    "entrypoint": "python -m determined.launch.deepspeed python3 main_timm.py",
    "name": args.experiment_name,
    "workspace": workspace_name,
    "project": project_name,
    "max_restarts": 2,
    "reproducibility": {"experiment_seed": 42},
    "resources": {"slots_per_trial": 2},
    "searcher": {
        "name": "single",
        "max_length": 1,
        "metric": "train_top1_acc",
        "smaller_is_better": False,
    },
    "environment": {"environment_variables": ["OMP_NUM_THREADS=1"]},
    "hyperparameters": {'ds_config': ds_config},
}



with suppress_stdout():
    client.create_experiment(config=config, model_dir=".")
