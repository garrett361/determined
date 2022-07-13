import argparse
from determined.experimental import client

import timm_models

parser = argparse.ArgumentParser(description="ImageNet Ensemble Loops")
parser.add_argument("-m", "--master", type=str, default="localhost:8080")
parser.add_argument("-nb", "--num_base_models", type=int, default=1)
parser.add_argument("-ne", "--num_ensembles", type=int, default=1)
parser.add_argument("-o", "--offset", type=int, default=0)
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("-d", "--dataset_name", type=str, default="imagenette2-160")
parser.add_argument("-st", "--skip_train", type=bool, default=True)
parser.add_argument("-es", "--ensemble_strategy", type=str, default="naive")
parser.add_argument("-mc", "--model_criteria", type=str, default="smallest")
parser.add_argument("-n", "--name", type=str, default="test")
parser.add_argument("-w", "--workspace", type=str, default="Ensembling")
parser.add_argument("-p", "--project", type=str, default="Tests")
parser.add_argument("-sc", "--sanity_check", type=bool, default=False)
args = parser.parse_args()

client.login(master=args.master, user="determined", password="")

config = {
    "entrypoint": "python -m determined.launch.torch_distributed -- python -m main",
    "name": args.name,
    "workspace": args.workspace,
    "project": args.project,
    "max_restarts": 0,
    "reproducibility": {"experiment_seed": 42},
    "resources": {"slots_per_trial": 1},
    "searcher": {"name": "single", "max_length": 1, "metric": "val_loss"},
    "hyperparameters": {
        "batch_size": args.batch_size,
        "dataset_name": args.dataset_name,
        "skip_train": args.skip_train,
        "ensemble_strategy": args.ensemble_strategy,
        "sanity_check": args.sanity_check,
    },
}

ensembles = timm_models.get_timm_ensembles_of_model_names(
    model_criteria=args.model_criteria,
    num_base_models=args.num_base_models,
    num_ensembles=args.num_ensembles,
    offset=args.offset,
)

for model_names in ensembles:
    config["hyperparameters"]["model_names"] = model_names
    client.create_experiment(config=config, model_dir=".")

print(
    "",
    80 * "*",
    f"Submitted {len(ensembles)} experiments to master at {args.master}",
    80 * "*",
    sep="\n",
)
