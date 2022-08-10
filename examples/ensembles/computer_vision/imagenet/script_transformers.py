#!/usr/bin/env python
from determined.experimental import client

import timm_models

MASTER = "http://104.196.135.13:8080"
USER = "determined"
PASSWORD = ""

client.login(master=MASTER, user=USER, password=PASSWORD)

MODEL_CRITERIA = "small"
NUM_BASE_MODELS = 1

model_names = timm_models.get_timm_ensembles_of_model_names(
    model_criteria=MODEL_CRITERIA,
    num_base_models=NUM_BASE_MODELS,
    num_ensembles=1,
)[0]

model_hparams = {
    "checkpoint_path_prefix": "shared_fs/state_dicts/",
    "model_names": model_names,
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
    "batches_per_epoch": None,
    "records_per_epoch": None,
}

data_hparams = {
    "dataset_name": "imagenette2-160",
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
