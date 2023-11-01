import logging
import random

import determined as det


def main(core_context) -> None:
    hps = det.get_cluster_info().trial.hparams
    metric_groups = ("val", "train", "test", "profile")
    for op in core_context.searcher.operations():
        for step in range(1, 6):
            for group in metric_groups:
                metric = random.random() * hps["value"] / step
                core_context.train.report_metrics(
                    group=group,
                    steps_completed=step,
                    metrics={f"{group}_metric": metric},
                )
        op.report_completed(metric)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
