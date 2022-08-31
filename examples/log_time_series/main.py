"""Minimal example of how one might log a time-series of externally-produced metrics and
pytorch state_dicts to determined. 
"""

from typing import Any, Dict

import determined as det
import torch
import torch.nn as nn


def create_fake_model_history(num_batches: int = 10) -> Dict[int, Any]:
    """Create a fake model history for testing purposes.  The reported metric values need to be
    serializable, hence the .item() calls
    """
    history = {
        batch_idx: {
            "metrics": {"loss": torch.randn(1).item(), "accuracy": torch.rand(1).item()},
            "state_dict": nn.Linear(3, 3).state_dict(),
        }
        for batch_idx in range(num_batches)
    }
    return history


if __name__ == "__main__":
    val_history = create_fake_model_history()
    with det.core.init() as core_context:
        for batch_idx, snapshot in val_history.items():
            steps_completed = batch_idx + 1
            metrics = snapshot["metrics"]
            state_dict = snapshot["state_dict"]
            core_context.train.report_validation_metrics(
                steps_completed=steps_completed, metrics=metrics
            )
            checkpoint_metadata = {"steps_completed": steps_completed}
            with core_context.checkpoint.store_path(checkpoint_metadata) as (path, storage_id):
                torch.save(state_dict, path.joinpath("state_dict.pth"))
