import logging
import time

import determined as det


def main(core_context) -> None:
    info = det.get_cluster_info()
    print(f"container_addrs: {info.container_addrs}")
    print(f"slot_ids: {info.slot_ids}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
