import logging

import determined as det


def main(core_context) -> None:
    for op in core_context.searcher.operations():
        for steps_completed in range(1, 11):
            if steps_completed % 2:
                core_context.train.report_training_metrics(
                    steps_completed=steps_completed, metrics={"loss": steps_completed}
                )
            else:
                core_context.train.report_validation_metrics(
                    steps_completed=steps_completed, metrics={"loss": steps_completed}
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
