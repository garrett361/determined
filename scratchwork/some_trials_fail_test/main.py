import logging
import time

import determined as det


def main(core_context) -> None:
    info = det.get_cluster_info()
    exit(info.trial.hparams["fail"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
