#!/usr/bin/env python3
from determined.experimental import client

MASTER = "http://127.0.0.1:8080"
USER = "determined"
PASSWORD = ""

client.login(master=MASTER, user=USER, password=PASSWORD)

model_hparams = {
    "n_filters1": 32,
    "n_filters2": 32,
    "dropout1": 0.25,
    "dropout2": 0.5,
}

optimizer_hparams = {"lr": 1e-4}

criterion_hparams = {}

trainer_hparams = {
    "worker_train_batch_size": 1024,
    "worker_val_batch_size": 2048,
    "train_metric_agg_rate": 8,
}

# max_length is in epochs
config = {
    "entrypoint": "python -m determined.launch.torch_distributed -- python3 main.py",
    "name": "mnist_pytorch_core_api",
    "max_restarts": 0,
    "reproducibility": {"experiment_seed": 42},
    "resources": {"slots_per_trial": 2},
    "searcher": {"name": "single", "max_length": 3, "metric": "val_loss"},
    "environment": {"environment_variables": ["OMP_NUM_THREADS=1"]},
    "hyperparameters": {
        "model": model_hparams,
        "optimizer": optimizer_hparams,
        "criterion": criterion_hparams,
        "trainer": trainer_hparams,
    },
}

client.create_experiment(config=config, model_dir=".")
