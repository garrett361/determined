import json
import logging
import random
import time

import determined as det
from determined.experimental import client


class CheckpointTester:
    def __init__(self, info, core_context) -> None:
        self.info = info
        self.core_context = core_context

        self.trial_id = self.info.trial.trial_id
        self.trial_reference = client.get_trial(self.trial_id)

        self.rank = core_context.distributed.rank
        self.local_rank = core_context.distributed.local_rank
        self.size = core_context.distributed.size
        self.is_distributed = self.size > 1
        self.is_chief = self.rank == 0
        self.is_local_chief = self.local_rank == 0

        if self.info.latest_checkpoint is None:
            self.steps_completed = 0
        else:
            with self.core_context.checkpoint.restore_path(info.latest_checkpoint) as path:
                with open(path.joinpath("metadata.json"), "r") as f:
                    metadata_dict = json.load(f)
                self.steps_completed = metadata_dict["steps_completed"]

    def save_checkpoint(self):
        self.steps_completed += 1
        self.core_context.train.report_validation_metrics(
            steps_completed=self.steps_completed, metrics={"arbitrary_metric_name": random.random()}
        )
        if self.is_chief:
            checkpoint_metadata = {
                "steps_completed": self.steps_completed,
            }
            with self.core_context.checkpoint.store_path(checkpoint_metadata) as (
                path,
                storage_id,
            ):
                with open(path.joinpath("test.txt"), "w") as f:
                    f.write("test")

    def select_checkpoint(self, *args, **kwargs):
        ckpt = self.trial_reference.select_checkpoint(*args, **kwargs)
        return ckpt


def main() -> None:
    info = det.get_cluster_info()
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        checkpoint_tester = CheckpointTester(info, core_context)
        # Create some random checkpoints.
        for _ in range(5):
            checkpoint_tester.save_checkpoint()
            checkpoint_tester.select_checkpoint(
                best=True,
                num_checkpoints="all",
                smaller_is_better=False,
                sort_by="arbitrary_metric_name",
            )

        # Get all checkpoints.
        all_checkpoints = checkpoint_tester.select_checkpoint(
            best=True,
            num_checkpoints="all",
            smaller_is_better=False,
            sort_by="arbitrary_metric_name",
        )

        logging.info("Deleting all checkpoints.")
        for ckpt in all_checkpoints:
            ckpt.delete()

        # Pause to ensure deletion.
        logging.info("Pausing to ensure deletion...")
        time.sleep(10)

        # Print the state of all checkpoints.
        # Need to re-load checkpoints because their state attr is defined upon construction.
        reloaded_checkpoints = checkpoint_tester.select_checkpoint(
            best=True,
            num_checkpoints="all",
            smaller_is_better=False,
            sort_by="arbitrary_metric_name",
        )
        logging.info(f"{[c.state for c in reloaded_checkpoints]}")

        # Wait a bit and try to delete them again.
        logging.info("Attempting to delete again.")
        time.sleep(5)
        for ckpt in reloaded_checkpoints:
            ckpt.delete()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    main()
