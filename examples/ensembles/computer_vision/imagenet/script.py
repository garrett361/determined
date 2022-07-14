import argparse

import attrdict
from determined.experimental import client

import timm_models
import workspaces

parser = argparse.ArgumentParser(description="ImageNet Ensemble Loops")
parser.add_argument("-m", "--master", type=str, default="localhost:8080")
parser.add_argument("-u", "--user", type=str, default="determined")
parser.add_argument("-p", "--password", type=str, default="")
parser.add_argument("-nb", "--num_base_models", type=int, default=1)
parser.add_argument("-ne", "--num_ensembles", type=int, default=1)
parser.add_argument("-o", "--offset", type=int, default=0)
parser.add_argument("-tb", "--train_batch_size", type=int, default=128)
parser.add_argument("-vb", "--val_batch_size", type=int, default=256)
parser.add_argument("-d", "--dataset_name", type=str, default="imagenette2-160")
parser.add_argument("-st", "--skip_train", type=bool, default=True)
parser.add_argument("-es", "--ensemble_strategy", type=str, default="naive")
parser.add_argument("-mc", "--model_criteria", type=str, default="smallest")
parser.add_argument("-n", "--name", type=str, default="test")
parser.add_argument("-w", "--workspace", type=str, default="Ensembling")
parser.add_argument("-sc", "--sanity_check", type=bool, default=True)
parser.add_argument("-t", "--test", type=bool, default=True)
args = parser.parse_args()

client.login(master=args.master, user=args.user, password=args.password)

config = attrdict.AttrDict(
    {
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
            "num_base_models": args.num_base_models,
        },
    }
)

workspace = workspaces.Workspace(
    workspace_name=config.workspace,
    master_url=args.master,
    username=args.user,
    password=args.password,
)
workspace.create_project(config.project)

ensembles = timm_models.get_timm_ensembles_of_model_names(
    model_criteria=args.model_criteria,
    num_base_models=args.num_base_models,
    num_ensembles=args.num_ensembles,
    offset=args.offset,
)

for model_names in ensembles:
    client.create_experiment(config=config, model_dir=".")
    print(f"Models in ensemble: {config.hyperparameters.model_names}")

print(
    "",
    80 * "*",
    f"Submitted {len(ensembles)} experiments to master at {args.master}",
    80 * "*",
    sep="\n",
)
