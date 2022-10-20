import json

import accelerate
import determined as det
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from attrdict import AttrDict
from determined.pytorch import TorchData
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, dimension: int, num_records: int) -> None:
        self.dimension = dimension
        self.num_records = num_records
        self.records = torch.randn(num_records, dimension)

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.records[idx]


def train_one_batch(batch: TorchData, model: nn.Module):
    output = model(batch)
    loss = F.mse_loss(output, batch)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":

    info = det.get_cluster_info()
    assert info is not None, "init_on_cluster() must be called on a Determined cluster."
    hparams = AttrDict(info.trial.hparams)
    latest_checkpoint = info.latest_checkpoint

    model = nn.Linear(hparams.dimension, hparams.dimension)
    dataset = RandomDataset(dimension=hparams.dimension, num_records=hparams.num_records)
    dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    accelerator = Accelerator(gradient_accumulation_steps=hparams.gradient_accumulation_steps)
    logger = accelerate.logging.get_logger(__name__)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    steps_completed = 0

    with det.core.init(distributed=distributed) as core_context:
        if latest_checkpoint is not None:
            with core_context.checkpoint.restore_path(latest_checkpoint) as path:
                with open(path.joinpath("metadata.json"), "r") as f:
                    checkpoint_metadata_dict = json.load(f)
                    steps_completed = checkpoint_metadata_dict["steps_completed"]

        # Emits a single operation of length max_len, as defined in the searcher config.
        for op in core_context.searcher.operations():
            while steps_completed < op.length:
                for batch in dataloader:
                    # Use the accumulate method for efficient gradient accumulation.
                    with accelerator.accumulate(model):
                        train_one_batch(batch, model)
                    took_sgd_step = accelerator.sync_gradients
                    if took_sgd_step:
                        steps_completed += 1
                        logger.info(f"Step {steps_completed} completed.")

                        is_end_of_training = steps_completed == op.length
                        time_to_report = steps_completed % metric_report_freq == 0
                        time_to_ckpt = steps_completed % checkpoint_freq == 0

                        # Report metrics, checkpoint, and preempt as appropriate.
                        if is_end_of_training or time_to_report or time_to_ckpt:
                            _report_train_metrics(core_context)
                            # report_progress for Web UI progress-bar rendering.
                            if accelerator.is_main_process:
                                op.report_progress(steps_completed)
                        if is_end_of_training or time_to_ckpt:
                            _save(core_context)
                            if core_context.preempt.should_preempt():
                                return
                        if is_end_of_training:
                            break
            if accelerator.is_main_process:
                # Report the final mean loss.
                op.report_completed(last_mean_loss)
