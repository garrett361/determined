import json
import logging
import random
from typing import Any, Generator, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from model import EmbedAndEncode, LMHead, Transformer, TransformerBlock
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

import determined as det

"""
Minimal transformer model FSDP script with Core API.
"""


def get_fake_data_iter(
    batch_size: int, vocab_size: int, max_seq_len: int, rank: int, device: torch.device
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Fake dataloader. Repeatedly yields the same (inputs, targets) tuple of tensors, with different
    tensors on different ranks.
    """
    torch.manual_seed(42 + rank)
    fake_sequence = torch.randint(vocab_size, (batch_size, max_seq_len), device=device)
    inputs, targets = fake_sequence[..., :-1], fake_sequence[..., 1:]
    while True:
        yield inputs, targets


def get_loss(fsdp_model: FSDP, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    inputs, labels = batch
    outputs = fsdp_model(inputs)
    outputs_flat = outputs.reshape(-1, outputs.shape[-1])
    labels_flat = labels.reshape(-1)
    loss = F.cross_entropy(outputs_flat, labels_flat)
    return loss


def get_reduced_loss_and_report(
    loss_history: list[torch.Tensor],
    steps_completed: int,
    core_context: det.core.Context,
) -> Optional[float]:
    """
    Average the most recent training losses across all processes and report the result. Returns the
    reduced loss on rank 0 and None on all other ranks.
    """

    loss_history_t = torch.stack(loss_history).mean()
    dist.reduce(loss_history_t, 0, op=dist.ReduceOp.AVG)
    if core_context.distributed.rank == 0:
        reduced_loss = loss_history_t.item()
        core_context.train.report_training_metrics(
            steps_completed=steps_completed, metrics={"loss": reduced_loss}
        )
        return reduced_loss


def save_checkpoint(
    fsdp_model: FSDP,
    optimizer: torch.optim.Optimizer,
    core_context: det.core.Context,
    steps_completed: int,
) -> None:
    # All ranks collectively build the checkpoint on rank 0:

    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state_dict = fsdp_model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(fsdp_model, optimizer)

    if core_context.distributed.rank == 0:
        with core_context.checkpoint.store_path(metadata={"steps_completed": steps_completed}) as (
            path,
            _,
        ):
            torch.save(model_state_dict, path.joinpath("model.bin"))
            torch.save(optim_state_dict, path.joinpath("optim.bin"))


def load_checkpoint(
    fsdp_model: FSDP,
    optimizer: torch.optim.Optimizer,
    core_context: det.core.Context,
    device: torch.device,
    uuid: str,
) -> int:
    with core_context.checkpoint.restore_path(uuid) as path:
        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            fsdp_model.load_state_dict(torch.load(path.joinpath("model.bin"), map_location=device))
            optim_state_dict = torch.load(path.joinpath("optim.bin"), map_location=device)
            optim_state_dict_to_load = FSDP.optim_state_dict_to_load(
                model=fsdp_model,
                optim=optimizer,
                optim_state_dict=optim_state_dict,
            )
            optimizer.load_state_dict(optim_state_dict_to_load)

        with open(path.joinpath("metadata.json"), "r") as f:
            metadata = json.load(f)

    last_step_completed = metadata["steps_completed"]
    return last_step_completed


def main(
    core_context: det.core.Context,
    hparams: dict[str, Any],
    checkpoint_uuid: Optional[str] = None,
) -> None:
    # Fix the random seed on all devices
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get and set the device for this process
    device = torch.device(f"cuda:{core_context.distributed.local_rank}")
    torch.cuda.set_device(device)

    # Build the unsharded model directly on the device.
    model = Transformer(
        d_model=hparams["d_model"],
        n_heads=hparams["n_heads"],
        vocab_size=hparams["vocab_size"],
        n_layers=hparams["n_layers"],
        max_seq_len=hparams["max_seq_len"],
        device=device,
    )

    # Inspect the model:
    if core_context.distributed.rank == 0:
        print("Model before FSDP:")
        print(model, flush=True)

    # Wrap the embedding layer, the lm head, and each transformer block into its own FSDP unit:
    auto_wrap_policy = ModuleWrapPolicy([TransformerBlock, EmbedAndEncode, LMHead])

    # The fsdp model:
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
    )

    # Inspect the model post-FSDP
    if core_context.distributed.rank == 0:
        print("Model after FSDP:")
        print(fsdp_model, flush=True)

    # The optimizer must be created post-FSDP
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=hparams["lr"])

    steps_completed = 0
    report_rate = hparams["report_rate"]
    checkpoint_rate = hparams["checkpoint_rate"]
    loss_history = []

    data_iter = get_fake_data_iter(
        batch_size=hparams["batch_size"],
        vocab_size=hparams["vocab_size"],
        max_seq_len=hparams["max_seq_len"],
        rank=core_context.distributed.rank,
        device=device,
    )

    # If a previous checkpoint exists, load it now and correct the steps_completed:
    if checkpoint_uuid is not None:
        steps_completed = load_checkpoint(
            fsdp_model, optimizer, core_context, device, checkpoint_uuid
        )

    for op in core_context.searcher.operations():
        # Train for the number of steps specified in searcher.max_length in config.yaml
        while steps_completed < op.length:
            batch = next(data_iter)
            loss = get_loss(fsdp_model, batch)
            loss_history.append(loss.detach().clone())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            steps_completed += 1
            this_is_the_last_step = steps_completed == op.length

            if steps_completed % report_rate == 0 or this_is_the_last_step:
                reduced_loss = get_reduced_loss_and_report(
                    loss_history, steps_completed, core_context
                )
                loss_history.clear()

            if steps_completed % checkpoint_rate == 0 or this_is_the_last_step:
                save_checkpoint(fsdp_model, optimizer, core_context, steps_completed)

        # Tell the master we're done
        if core_context.distributed.rank == 0:
            op.report_completed(reduced_loss)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    assert info, "This script must run on a determined cluster."
    assert torch.cuda.is_available(), "This script assumes cuda."

    checkpoint_uuid = info.latest_checkpoint
    hparams = info.trial.hparams
    try:
        dist.init_process_group("nccl")
        distributed = det.core.DistributedContext.from_torch_distributed()
        with det.core.init(distributed=distributed) as core_context:
            main(
                core_context=core_context,
                hparams=hparams,
                checkpoint_uuid=checkpoint_uuid,
            )
    finally:
        dist.destroy_process_group()
