"""Minimal example demonstrating the lack of metrics in the Web UI when using Core API and 0.19.3"""
import logging
import time

import determined as det
import torch

logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)


def main() -> None:
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    steps_completed = 0
    with det.core.init(distributed=distributed) as core_context:
        # We are emitting a single op of length max_length, as set in const.yaml.
        for op in core_context.searcher.operations():
            while steps_completed < op.length:
                steps_completed += 1
                # Generate some fake metrics.
                loss = 1 / steps_completed
                # Report training metrics every step.
                core_context.train.report_training_metrics(
                    steps_completed=steps_completed,
                    metrics={"loss": loss * torch.randn(1).exp().item()},
                )
                # Report validation metrics every ten steps.
                if not steps_completed % 10:
                    core_context.train.report_validation_metrics(
                        steps_completed=steps_completed,
                        metrics={"loss": loss * torch.randn(1).exp().item()},
                    )
                time.sleep(1)
            op.report_completed(loss)


if __name__ == "__main__":
    main()
