import logging
import time
from typing import Optional

from tester_class import CheckpointTester

import determined as det

"""
Dynamically delete all but the three best checkpoints during 'training'. The config's gc
policy is set to instead keep up to ten checkpoints.
"""


def main() -> None:
    info = det.get_cluster_info()
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        checkpoint_tester = CheckpointTester(info, core_context, "arbitrary_metric_name")

        # Create random checkpoints, gc-ing
        for _ in range(9):
            checkpoint_tester.save_checkpoint()
            time.sleep(3)
            checkpoint_tester.delete_worst_checkpoints(num_to_keep=3)
            time.sleep(3)
            num_non_deleted_checkpoints = len(checkpoint_tester.get_all_best_sorted_checkpoints())
            logging.info(f"Number of non-deleted checkpoints: {num_non_deleted_checkpoints}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    main()
