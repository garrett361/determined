#!/usr/bin/env python
from determined.experimental import client

MASTER = "http://104.196.135.13:8080"
USER = "determined"
PASSWORD = ""

client.login(master=MASTER, user=USER, password=PASSWORD)

model_hparams = {}

optimizer_hparams = {"lr": 0.001}

trainer_hparams = {
    "worker_train_batch_size": 256,
    "worker_val_batch_size": 512,
    "train_metric_agg_rate": 16,
}

# max_length is in epochs
config = {
    "entrypoint": "python -m determined.launch.torch_distributed -- python3 main_transformers.py",
    "workspace": "Test",
    "project": "test",
    "name": "transformer_test",
    "max_restarts": 0,
    "reproducibility": {"experiment_seed": 42},
    "resources": {"slots_per_trial": 1},
    "searcher": {"name": "single", "max_length": 3, "metric": "val_loss"},
    "environment": {"environment_variables": ["OMP_NUM_THREADS=1"]},
    "hyperparameters": {
        "model": model_hparams,
        "optimizer": optimizer_hparams,
        "trainer": trainer_hparams,
    },
}

client.create_experiment(config=config, model_dir=".")
