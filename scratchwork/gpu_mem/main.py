import logging

import torch

import determined as det


def main(core_context) -> None:
    gpu_mem = int(torch.cuda.get_device_properties(0).total_memory)
    core_context.train.report_training_metrics(steps_completed=1, metrics={"gpu_mem": gpu_mem})
    core_context.train.report_validation_metrics(steps_completed=1, metrics={"gpu_mem": gpu_mem})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
