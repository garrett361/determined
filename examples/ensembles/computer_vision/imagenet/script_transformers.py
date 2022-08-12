#!/usr/bin/env python
from determined.experimental import client

import timm_models

GCP_T4_MASTER = "http://104.196.135.13:8080"
GCP_V100_MASTER = "http://34.148.22.184:8080"
MASTER = GCP_V100_MASTER
USER = "determined"
PASSWORD = ""

client.login(master=MASTER, user=USER, password=PASSWORD)

MODEL_CRITERIA = "small"
NUM_BASE_MODELS = 3

model_names = timm_models.get_timm_ensembles_of_model_names(
    model_criteria=MODEL_CRITERIA,
    num_base_models=NUM_BASE_MODELS,
    num_ensembles=1,
)[0]

model_hparams = {
    "checkpoint_path_prefix": "shared_fs/state_dicts/",
    "model_names": model_names,
    "num_layers": 1,
    "num_heads": 2,
    "dim_feedforward": 1024,
    "mix_models": False,
    "mix_classes": False,
}


optimizer_hparams = {"lr": {"type": "log", "base": 10, "minval": -6, "maxval": -3, "count": 16}}

trainer_hparams = {
    "worker_train_batch_size": 256,
    "worker_val_batch_size": 256,
    "train_metric_agg_rate": 4,
    "max_len_unit": "epochs",  # TODO: doesn't do anything yet
    "batches_per_epoch": None,  # TODO: doesn't do anything yet
    "records_per_epoch": None,  # TODO: doesn't do anything yet
}

data_hparams = {
    "dataset_name": "imagenetv2-top-images",
}

# max_length is in epochs
max_epochs = 16

searcher_config = {
    "name": "grid",
    "max_length": max_epochs,
    "metric": "val_top1_acc",  # TODO: Also currently hard-coded into the Trainer. Change.
    "smaller_is_better": False,
}


config = {
    "entrypoint": "python -m determined.launch.torch_distributed -- python3 main_transformers.py",
    "workspace": "Test",
    "project": "Test",
    "name": f"transformer_test_{NUM_BASE_MODELS}_{data_hparams['dataset_name']}",
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
