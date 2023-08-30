import logging

import determined as det


def main(core_context) -> None:
    metadata = {"steps_completed": 0}
    with core_context.checkpoint.store_path(metadata) as _:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
