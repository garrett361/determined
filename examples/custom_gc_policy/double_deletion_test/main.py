import logging
import time

from tester_class import CheckpointTester

import determined as det


def main() -> None:
    info = det.get_cluster_info()
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        checkpoint_tester = CheckpointTester(info, core_context, "arbitrary_metric_name")
        # Create some random checkpoints.
        for _ in range(5):
            checkpoint_tester.save_checkpoint()
            checkpoint_tester.select_checkpoint(
                best=True,
                num_checkpoints="all",
                smaller_is_better=False,
                sort_by=checkpoint_tester.searcher_metric,
            )

        # Get all checkpoints.
        all_checkpoints = checkpoint_tester.select_checkpoint(
            best=True,
            num_checkpoints="all",
            smaller_is_better=False,
            sort_by=checkpoint_tester.searcher_metric,
        )

        logging.info("Deleting all checkpoints.")
        for ckpt in all_checkpoints:
            ckpt.delete()

        # Pause to ensure deletion.
        logging.info("Pausing to ensure deletion...")
        time.sleep(10)

        # Verify all checkpoints are deleted.
        # Need to re-load checkpoints because their state attr is defined upon construction.
        reloaded_checkpoints = checkpoint_tester.select_checkpoint(
            best=True,
            num_checkpoints="all",
            smaller_is_better=False,
            sort_by="arbitrary_metric_name",
        )
        reloaded_checkpoint_states = [c.state for c in reloaded_checkpoints]
        assert all(
            [s.name == "DELETED" for s in reloaded_checkpoint_states]
        ), "Some checkpoints were not deleted!"
        logging.info(f"All checkpoints successfully deleted.")

        # Try to delete them again and check whether it errors.
        logging.info("Attempting to delete again.")
        for ckpt in reloaded_checkpoints:
            ckpt.delete()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    main()
