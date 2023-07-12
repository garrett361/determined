import logging
import os

import determined as det


def main(core_context) -> None:
    logging.info(str(os.environ))
    assert os.environ["MY_CUSTOM_ENV_VAR"] == "value"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_deepspeed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
