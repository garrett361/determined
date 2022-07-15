import argparse

import attrdict
from determined.experimental import client

import timm_models
import workspaces

parser = argparse.ArgumentParser(description="ImageNet Ensemble Loops")
parser.add_argument("-m", "--master", type=str, default="localhost:8080")
parser.add_argument("-u", "--user", type=str, default="determined")
parser.add_argument("-p", "--password", type=str, default="")
parser.add_argument("-d", "--dataset_name", type=str, default="imagenette2-160")
parser.add_argument("-es", "--ensemble_strategy", type=str, default="")
parser.add_argument("-mc", "--model_criteria", type=str, default="")
parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("-w", "--workspace", type=str, default="Ensembling")
parser.add_argument("-mn", "--model_names", nargs="+", type=str, default=[])
parser.add_argument("-nb", "--num_base_models", nargs="+", type=int, default=[])
parser.add_argument("-cpp", "--checkpoint_path_prefix", type=str, default="shared_fs/state_dicts/")
parser.add_argument("-ne", "--num_ensembles", type=int, default=0)
parser.add_argument("-o", "--offset", type=int, default=0)
parser.add_argument("-tb", "--train_batch_size", type=int, default=128)
parser.add_argument("-vb", "--val_batch_size", type=int, default=256)
parser.add_argument("-st", "--skip_train", action="store_true")
parser.add_argument("-sc", "--sanity_check", action="store_true")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("-lm", "--list_models", action="store_true")
args = parser.parse_args()

if args.model_names and (args.num_base_models or args.num_ensembles or args.model_criteria):
    raise ValueError(
        "Setting --model_names is mutually exclusive with setting --num_base_models,"
        " --num_ensembles, or --model_criteria"
    )
if args.model_names:
    args.num_base_models = [len(args.model_names)]
    args.num_ensembles = 1

# Generate name from command line arguments, if none provided.
generate_names = args.name == ""
if args.list_models:
    print(timm_models.get_all_model_names())

client.login(master=args.master, user=args.user, password=args.password)
print(
    80 * "-",
    f"Submitting {len(args.num_base_models) * args.num_ensembles} experiment(s).",
    80 * "-",
    sep="\n",
)

for num_base_models in args.num_base_models:
    if generate_names:
        name_components = [
            f"{'manual' if args.model_names else args.model_criteria}",
            f"{args.ensemble_strategy}",
            f"{num_base_models}",
        ]
        args.name = "_".join(name_components)

    config = {
        "entrypoint": "python -m determined.launch.torch_distributed -- python -m main",
        "name": args.name,
        "workspace": args.workspace + ("_test" if args.test else ""),
        "project": args.dataset_name,
        "max_restarts": 0,
        "reproducibility": {"experiment_seed": 42},
        "resources": {"slots_per_trial": 1},
        "searcher": {"name": "single", "max_length": 1, "metric": "val_loss"},
        "environment": {"environment_variables": ["OMP_NUM_THREADS=1"]},
        "hyperparameters": {
            "train_batch_size": args.train_batch_size,
            "val_batch_size": args.val_batch_size,
            "dataset_name": args.dataset_name,
            "skip_train": args.skip_train,
            "ensemble_strategy": args.ensemble_strategy,
            "model_criteria": args.model_criteria,
            "sanity_check": args.sanity_check,
            "num_base_models": num_base_models,
            "checkpoint_path_prefix": args.checkpoint_path_prefix,
        },
    }

    workspace = workspaces.Workspace(
        workspace_name=config["workspace"],
        master_url=args.master,
        username=args.user,
        password=args.password,
    )
    workspace.create_project(config["project"])

    if args.model_names:
        ensembles = [args.model_names]
    else:
        ensembles = timm_models.get_timm_ensembles_of_model_names(
            model_criteria=args.model_criteria,
            num_base_models=num_base_models,
            num_ensembles=args.num_ensembles,
            offset=args.offset,
        )

    for model_names in ensembles:
        config["hyperparameters"]["model_names"] = model_names
        client.create_experiment(config=config, model_dir=".")
        print(
            f"{num_base_models} models in ensemble: {config['hyperparameters']['model_names']}",
            "\n",
        )
