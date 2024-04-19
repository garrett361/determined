import logging
import os

import determined as det


def main(core_context) -> None:
    rank = os.environ["RANK"]
    metadata = {
        "steps_completed": 0,
    }
    with core_context.checkpoint.store_path(metadata, shard=True) as (path, uuid):
        with open(path / "conflict", "w") as f:
            f.write(rank)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    distributed = det.core.DistributedContext.from_torch_distributed()
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
