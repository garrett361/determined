#!/usr/bin/env python
from determined.experimental import client

MASTER = "http://34.148.22.184:8080"
USER = "determined"
PASSWORD = ""

client.login(master=MASTER, user=USER, password=PASSWORD)

model_hparams = {
    "checkpoint_path_prefix": "shared_fs/state_dicts/",
    "num_base_models": 3,
    "model_criteria": "small",
    "num_layers": 3,
    "num_heads": 4,
    "dim_feedforward": 2048,
    "mix_models": True,
    "mix_classes": True,
}

optimizer_hparams = {"lr": 0.001}

trainer_hparams = {
    "worker_train_batch_size": 256,
    "worker_val_batch_size": 256,
    "train_metric_agg_rate": 4,
    "max_len_unit": "epochs",  # TODO: doesn't do anything yet
}

data_hparams = {
    "name": "imagenette2-160",
}


# max_length is in epochs
max_epochs = 1

searcher_config = {
    "name": "single",
    "max_length": max_epochs,
    "metric": "val_top1_acc",  # TODO: Also needs to be hard-coded into the Trainer. Change.
    "smaller_is_better": False,
}


config = {
    "entrypoint": "python -m determined.launch.torch_distributed -- python3 main_transformers.py",
    "workspace": "Test",
    "project": "Test",
    "name": "transformer_test",
    "max_restarts": 0,
    "reproducibility": {"experiment_seed": 42},
    "resources": {"slots_per_trial": 1},
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
