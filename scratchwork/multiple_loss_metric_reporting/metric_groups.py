import logging
import random

import determined as det

metric_names = ["loss", "t_loss", "val_loss", "v_loss"]


def main(core_context: det.core.Context):
    for step in range(1, 11):
        metrics = {name: (step * random.random()) ** 2 for name in metric_names}
        core_context.train.report_metrics(group="training", steps_completed=step, metrics=metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format=det.LOG_FORMAT)
    with det.core.init() as core_context:
        main(core_context=core_context)
