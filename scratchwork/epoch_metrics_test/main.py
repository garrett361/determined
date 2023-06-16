import logging
import time

import determined as det


def main(core_context) -> None:
    for op in core_context.searcher.operations():
        for epoch in range(1, 11):
            core_context.train.report_training_metrics(
                steps_completed=epoch, metrics={"loss": epoch, "epoch": epoch}
            )
            core_context.train.report_validation_metrics(
                steps_completed=epoch, metrics={"loss": epoch, "epoch": epoch}
            )
        op.report_completed(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
