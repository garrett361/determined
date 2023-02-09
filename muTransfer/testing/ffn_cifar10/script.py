import argparse
import contextlib
import copy
import itertools
import json
import os
import sys

import pandas as pd
import workspaces
from determined.experimental import client


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


MAX_LENGTH = 10
NAME = "ffn_cifar10"

DEFAULT_CONFIG = {
    "name": NAME,
    "max_restarts": 5,
    "resources": {"slots_per_trial": 1, "max_slots": 8},
    "environment": {
        "environment_variables": ["OMP_NUM_THREADS=1"],
    },
    "searcher": {
        "name": "grid",
        "metric": "loss",
        "max_length": MAX_LENGTH,
    },  # In units of distributed batches.
    # None fields are populated by argparse.
    "hyperparameters": {
        "use_mutransfer": None,
        "optimizer_name": None,
        "random_seed": 42,
        "trainer": {"batch_size": 512, "metric_agg_rate": MAX_LENGTH // 10},
        "model": {
            "num_hidden_layers": None,
            "input_dim": 3 * 32 * 32,
            "output_dim": 10,
            "width_multiplier": None,
        },
        "optimizer": {"lr": {"type": "log", "base": 10, "minval": -4, "maxval": 0, "count": 10}},
    },
    "entrypoint": f"python3 -m determined.launch.torch_distributed python3 -m {NAME}.main",
}


def parse_args():

    parser = argparse.ArgumentParser(description="HF GLUE DeepSpeedTrainer Loops")
    parser.add_argument("-u", "--user", type=str, default="determined")
    parser.add_argument("-p", "--password", type=str, default="")
    parser.add_argument("-slots", "--slots_per_trial", type=int, nargs="+", default=[1])
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-w", "--workspace", type=str, default="muTransfer")
    parser.add_argument("-on", "--optimizer_name", type=str, nargs="+", default=["sgd"])
    parser.add_argument("-wm", "--width_multiplier", type=int, nargs="+", default=[1])
    parser.add_argument("-nhl", "--num_hidden_layers", type=int, nargs="+", default=[3])
    parser.add_argument("-ens", "--exp_name_suffix", type=str, nargs="+", default=[""])
    parser.add_argument("-ad", "--allow_duplicates", action="store_true")
    parser.add_argument("-nm", "--no_mutransfer", action="store_true")

    parser.add_argument("-t", "--test", action="store_true")
    args = parser.parse_args()
    return args


def exp_name_and_config_generator(args):
    """
    Loops over provided autotuning_tuner_type and searcher_metric values and yields
    exp_name, config tuples.
    """
    # Various sanity checks, hacks, and dynamic fixes.
    # Generate experiment names from command line arguments
    # Append '_test' to the given workspace name, if --test is set.
    workspace_name = args.workspace + ("_test" if args.test else "")

    with suppress_stdout():
        client.login(user=args.user, password=args.password)

    workspace = workspaces.Workspace(
        workspace_name=workspace_name,
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
        for _, vals in existing_trials_df.iterrows():
            try:
                existing_exp_names.add(vals.exp_name)
            except AttributeError:
                pass

    for (
        optimizer_name,
        width_multiplier,
        slots_per_trial,
        num_hidden_layers,
        suffix,
    ) in itertools.product(
        args.optimizer_name,
        args.width_multiplier,
        args.slots_per_trial,
        args.num_hidden_layers,
        args.exp_name_suffix,
    ):
        # Dynamically generate project and experiment names.
        if args.project_name:
            project_name = args.project_name
        else:
            project_name = DEFAULT_CONFIG["name"]
            if project_name not in existing_project_names:
                existing_project_names.append(project_name)
                workspace.create_project(project_name)

        exp_name = f"{DEFAULT_CONFIG['name']}.{DEFAULT_CONFIG['searcher']['name']}"
        exp_name += (
            f".{slots_per_trial}GPU.{width_multiplier}wm.{num_hidden_layers}hl.{optimizer_name}"
        )
        if not args.no_mutransfer:
            exp_name += ".use_mutransfer"
        if suffix:
            exp_name += f".{suffix}"

        if exp_name in existing_exp_names:
            print(f"Skipping {exp_name}: already exists.")
            continue

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["workspace"] = workspace_name
        config["project"] = project_name
        config["name"] = exp_name
        config["hyperparameters"]["exp_name"] = exp_name  # Added for easy duplication checking.
        config["hyperparameters"]["optimizer_name"] = optimizer_name
        config["hyperparameters"]["use_mutransfer"] = not args.no_mutransfer
        config["resources"]["slots_per_trial"] = slots_per_trial
        config["hyperparameters"]["model"]["width_multiplier"] = width_multiplier
        config["hyperparameters"]["model"]["num_hidden_layers"] = num_hidden_layers
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
