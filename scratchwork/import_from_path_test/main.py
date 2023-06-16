import logging
import os

import determined as det


def main(core_context) -> None:
    from new_code import model

    with det.import_from_path("old_code"):
        import model as old_model
    assert model.name == "new"
    assert old_model.name == "old"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
