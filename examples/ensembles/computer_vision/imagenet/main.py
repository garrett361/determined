import logging

import determined as det

import data
import ensemble_trainer
import timm_models


def main(core_context) -> None:
    name_ensembles = timm_models.get_timm_ensembles_of_model_names(
        criteria="smallest", num_base_models=2, num_ensembles=1
    )
    ensemble_strategy = "naive"
    print(f"Building train_dataset")
    train_dataset = None  # data.get_dataset(name="imagenette2-160", split="train")
    print(f"Building val_dataset")
    val_dataset = data.get_dataset(name="imagenette2-160", split="val")
    for ensemble in name_ensembles:
        val_logging_data = {"model_names": ensemble, "ensemble_strategy": ensemble_strategy}
        model_list = timm_models.build_timm_model_list(ensemble, pretrained=True)
        trainer = ensemble_trainer.EnsembleTrainer(
            core_context,
            model_list=model_list,
            batch_size=128,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            ensemble_strategy="naive",
            ensemble_args=None,
            val_logging_data=val_logging_data,
        )
        trainer.build_ensemble()
        trainer.validate_ensemble()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
