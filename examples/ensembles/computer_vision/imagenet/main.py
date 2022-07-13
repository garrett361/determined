import logging

import determined as det

import attrdict
import data
import ensemble_trainer
import timm_models


def main(core_context, hparams) -> None:
    if hparams.skip_train:
        print(f"Skipping building train_dataset")
        train_dataset = None
    else:
        print(f"Building train_dataset")
        train_dataset = data.get_dataset(name=hparams.dataset_name, split="train")

    print(f"Building val_dataset")
    val_dataset = data.get_dataset(name=hparams.dataset_name, split="val")
    val_logging_data = {
        "model_names": hparams.model_names,
        "ensemble_strategy": hparams.ensemble_strategy,
        "batch_size": hparams.batch_size,
        "dataset_name": hparams.dataset_name,
    }
    model_list = timm_models.build_timm_model_list(hparams.model_names, pretrained=True)
    trainer = ensemble_trainer.EnsembleTrainer(
        core_context,
        model_list=model_list,
        batch_size=hparams.batch_size,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        ensemble_strategy=hparams.ensemble_strategy,
        ensemble_args=None,
        val_logging_data=val_logging_data,
        sanity_check=hparams.sanity_check,
    )
    trainer.build_ensemble()
    trainer.validate_ensemble()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    hparams = attrdict.AttrDict(info.trial.hparams)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, hparams)
