import json
import logging
import random
import time
from typing import Optional

import determined as det
from determined.experimental import client


class CheckpointTester:
    def __init__(self, info, core_context) -> None:
        self.info = info
        self.core_context = core_context

        self.trial_id = self.info.trial.trial_id
        self.trial_reference = client.get_trial(self.trial_id)
        self.smaller_is_better = self.info.trial._config["searcher"]["smaller_is_better"]

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

    def get_all_best_sorted_checkpoints(self, include_deleted: bool = False):
        checkpoints = self.select_checkpoint(
            best=True,
            num_checkpoints="all",
            smaller_is_better=self.smaller_is_better,
            sort_by="arbitrary_metric_name",
        )
        if not include_deleted:
            checkpoints = [c for c in checkpoints if c.state.name != "DELETED"]
        return checkpoints

    def delete_worst_checkpoints(
        self, num_to_keep: Optional[int] = None, num_to_delete: Optional[int] = None
    ):
        best_checkpoints = self.get_all_best_sorted_checkpoints()
        assert not (num_to_keep is not None and num_to_delete is not None)
        if num_to_keep is not None:
            checkpoints_to_delete = best_checkpoints[num_to_keep:]
        elif num_to_delete is not None:
            checkpoints_to_delete = best_checkpoints[-num_to_delete:]

        for c in checkpoints_to_delete:
            c.delete()


def main() -> None:
    info = det.get_cluster_info()
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        checkpoint_tester = CheckpointTester(info, core_context)

        # Create random checkpoints, gc-ing
        for _ in range(9):
            checkpoint_tester.save_checkpoint()
            time.sleep(2)
            print(checkpoint_tester.get_all_best_sorted_checkpoints())
            checkpoint_tester.delete_worst_checkpoints(num_to_keep=3)
            time.sleep(2)
            num_non_deleted_checkpoints = len(checkpoint_tester.get_all_best_sorted_checkpoints())
            logging.info(f"Number of non-deleted checkpoints: {num_non_deleted_checkpoints}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    main()
