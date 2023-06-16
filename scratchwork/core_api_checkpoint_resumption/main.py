import logging

import determined as det

info = det.get_cluster_info()
logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)


def main(core_context, info):
    latest_checkpoint = info.latest_checkpoint
    trial_id = info.trial.trial_id
    logging.info(f"Latest checkpoint: {latest_checkpoint}")

    if latest_checkpoint is not None:
        with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            with open(path / "state_dict.pt", "r") as f:
                print(f.read())

    metadata = {"steps_completed": 0}
    with core_context.checkpoint.store_path(metadata) as (path, uuid):
        with open(path / "state_dict.pt", "w") as f:
            f.write(f"This is a checkpoint for {trial_id}")


if __name__ == "__main__":
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, info)
