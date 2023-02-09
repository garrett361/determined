import json
import random
from typing import Optional

from determined.experimental import client


class CheckpointTester:
    def __init__(self, info, core_context, searcher_metric) -> None:
        self.info = info
        self.core_context = core_context
        self.searcher_metric = searcher_metric

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
            steps_completed=self.steps_completed, metrics={self.searcher_metric: random.random()}
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
            sort_by=self.searcher_metric,
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

        if self.is_chief:
            for c in checkpoints_to_delete:
                c.delete()
