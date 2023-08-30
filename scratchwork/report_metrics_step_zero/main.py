import logging

import determined as det


def main(core_context) -> None:
    core_context.train.report_training_metrics(steps_completed=0, metrics={"some_train_metric": 0})
    core_context.train.report_validation_metrics(steps_completed=0, metrics={"some_val_metric": 1})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
