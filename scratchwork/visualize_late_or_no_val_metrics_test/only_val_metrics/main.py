import logging
import random
import time

import determined as det


def main(core_context) -> None:
    for steps_completed in range(10):
        core_context.train.report_validation_metrics(
            steps_completed=steps_completed, metrics={"train_loss": random.random()}
        )
        time.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
