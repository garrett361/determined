import logging
import random

import determined as det

metric_groups = ["inference", "test"]
metric_names = [
    "acc",
    "f1",
    "this_is_a_super_long_name_with_underscores_everywhere",
    "this is a very long name with spaces instead of underscores",
]

random.seed(2)


def main(core_context: det.core.Context):
    for step in range(10):
        for group in metric_groups:
            metrics = {name: (step * random.random()) ** 2 for name in metric_names}
            core_context.train.report_metrics(group=group, steps_completed=step, metrics=metrics)

    core_context.train.report_validation_metrics(steps_completed=10, metrics=metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format=det.LOG_FORMAT)
    with det.core.init() as core_context:
        main(core_context=core_context)
