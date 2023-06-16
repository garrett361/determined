import logging
import time

import determined as det


def main(core_context) -> None:
    info = det.get_cluster_info()
    rendezvous_info = {k: str(v) for k, v in info._rendezvous_info.__dict__.items()}
    logging.info(str(rendezvous_info))
    # Turn keys which are stringified lists back into lists.
    container_addrs = rendezvous_info["container_addrs"]
    container_slot_counts = rendezvous_info["container_slot_counts"]
    container_addrs = container_addrs.strip("[]'").split(",")
    container_slot_counts = container_slot_counts.strip("[]'").split(",")

    # Should always be submitting with one GPU slot.
    assert len(container_slot_counts) == len(container_addrs) == 1
    # Turn both into ints for Web UI purposes (apparently we don't display strings?).
    metrics = {
        "container_addrs_suffix": int(container_addrs.pop().split(".")[-1]),
        "container_slot_counts": int(container_slot_counts[0]),
    }
    # Use the last few digits of the current time in seconds as the step number.
    steps_completed = int(str(int(time.time()))[-5:])
    core_context.train.report_validation_metrics(steps_completed=steps_completed, metrics=metrics)
    time.sleep(10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
