import logging
import random

import determined as det
from determined.common import util as det_util

metric_groups = [det_util._LEGACY_TRAINING, det_util._LEGACY_VALIDATION, "inference", "test"]
metric_names = ["acc", "f1"]


def main(core_context: det.core.Context):
    for step in range(10):
        for group in metric_groups:
            metrics = {name: (step * random.random()) ** 2 for name in metric_names}
            core_context.train._report_trial_metrics(
                group=group, total_batches=step, metrics=metrics
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format=det.LOG_FORMAT)
    with det.core.init() as core_context:
        main(core_context=core_context)
