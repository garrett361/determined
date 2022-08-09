#!/usr/bin/env python
from determined.experimental import client

MASTER = "http://104.196.135.13:8080"
USER = "determined"
PASSWORD = ""

client.login(master=MASTER, user=USER, password=PASSWORD)

model_hparams = {
    "checkpoint_path_prefix": "shared_fs/state_dicts/",
    "num_base_models": 3,
    "model_criteria": "small",
}

optimizer_hparams = {"lr": 0.001}

trainer_hparams = {
    "worker_train_batch_size": 128,
    "worker_val_batch_size": 256,
    "train_metric_agg_rate": 2,
}

data_hparams = {
    "name": "imagenette2-160",
}


# max_length is in epochs
max_epochs = 5

searcher_config = {
    "name": "single",
    "max_length": max_epochs,
    "metric": "val_top1_acc",  # TODO: Also needs to be hard-coded into the Trainer. Change.
    "smaller_is_better": False,
}


config = {
    "entrypoint": "python -m determined.launch.torch_distributed -- python3 main_transformers.py",
    "workspace": "Test",
    "project": "test",
    "name": "transformer_test",
    "max_restarts": 0,
    "reproducibility": {"experiment_seed": 42},
    "resources": {"slots_per_trial": 4},
    "searcher": searcher_config,
    "environment": {"environment_variables": ["OMP_NUM_THREADS=1"]},
    "hyperparameters": {
        "model": model_hparams,
        "optimizer": optimizer_hparams,
        "trainer": trainer_hparams,
        "data": data_hparams,
    },
}


client.create_experiment(config=config, model_dir=".")
