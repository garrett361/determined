import argparse
import gc
import json
import logging
import random
from typing import Any, Callable, Dict, Generator, List, Literal, Tuple, Union

import determined as det
import deepspeed
import numpy as np
import torch
import torch.nn as nn
from determined.pytorch import TorchData

import data

from constants import DS_CONFIG_PATH


class DeepSpeedTrainer(nn.Module):
    def __init__(
        self,
        core_context: det.core.Context,
        latest_checkpoint: str,
        args: argparse.Namespace,
        model: nn.Module,
        transforms: Union[Callable, List[Callable]],
        dataset_name: str,
        sanity_check: bool = False,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.core_context = core_context
        self.latest_checkpoint = latest_checkpoint
        self.args = args
        self.model = model
        self.transforms = transforms
        self.dataset_name = dataset_name
        self.sanity_check = sanity_check
        if self.sanity_check:
            logging.info(f"Running in sanity check mode!")
        self.random_seed = random_seed

        self.criterion = nn.CrossEntropyLoss()

        self.rank = core_context.distributed.rank
        self.is_chief = self.rank == 0
        self.local_rank = core_context.distributed.local_rank
        self.is_local_chief = self.local_rank == 0

        self.steps_completed = 0

        # Instantiated as needed through private methods.
        self.train_dataset = None
        self.train_loader = None
        self.model_engine = None
        self.optimizer = None
        self.fp16 = None
        self.device = None

        self._setup()

    def _set_random_seeds(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.random.manual_seed(self.random_seed)

    def _build_datasets(self) -> None:
        self.train_dataset = data.get_dataset(
            dataset_name=self.dataset_name, split="train", transforms=self.transforms
        )

    def _deepspeed_init(self) -> None:
        deepspeed.init_distributed()
        self.model_engine, self.optimizer, self.train_loader, __ = deepspeed.initialize(
            args=self.args,
            model=self.model,
            model_parameters=self.model.parameters(),
            training_data=self.train_dataset,
        )
        self.fp16 = self.model_engine.fp16_enabled()
        # DeepSpeed uses the local_rank as the device, for some reason.
        self.device = self.model_engine.device

    def _setup(self) -> None:
        self._build_datasets()
        self._deepspeed_init()

    def _batch_generator(
        self,
        split: Literal["train", "val", "test"],
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        loader_dict = {
            "train": self.train_loader,
        }
        loader = loader_dict[split]
        for batch in loader:
            inputs, targets = batch
            if self.fp16:
                inputs = inputs.half()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            yield inputs, targets

    def _train_one_batch(self, batch_idx: int, inputs: TorchData, targets: TorchData) -> None:
        outputs = self.model_engine(inputs)
        loss = self.criterion(outputs, targets)
        self.model_engine.backward(loss)
        self.model_engine.step()

    def train(self) -> None:
        if self.latest_checkpoint is not None:
            self._restore()
        for op in self.core_context.searcher.operations():
            while self.steps_completed < op.length:
                for batch_idx, (inputs, targets) in enumerate(self._batch_generator(split="train")):
                    self._train_one_batch(batch_idx=batch_idx, inputs=inputs, targets=targets)
                self.steps_completed += 1
                if self.is_chief:
                    op.report_progress(self.steps_completed)
                if self.core_context.preempt.should_preempt():
                    return
            if self.is_chief:
                op.report_completed(0)

    @staticmethod
    def update_tmbspg_in_config(
        train_micro_batch_size_per_gpu: int, path: str = DS_CONFIG_PATH
    ) -> None:
        with open(path, "r") as f:
            old_config = json.load(f)
        old_config["train_micro_batch_size_per_gpu"] = train_micro_batch_size_per_gpu
        with open(path, "w") as f:
            json.dump(old_config, f)

    @classmethod
    def find_max_batch_size(
        cls, max_fails: int = 3, initial_batch_size: int = 1, batches_to_train: int = 2, **kwargs
    ) -> int:
        lo, hi = 1, 2 * initial_batch_size
        fails = 0
        batch_size = 0
        while fails <= max_fails:
            mid = (lo + hi) // 2
            trainer = None
            core_context = kwargs["core_context"]
            is_chief = core_context.distributed.rank == 0
            try:
                if is_chief:
                    cls.update_tmbspg_in_config(train_micro_batch_size_per_gpu=mid)
                kwargs["args"].train_micro_batch_size_per_gpu = mid
                core_context.distributed.broadcast(None)  # Hack for syncing.
                trainer = cls(**kwargs)
                # Instantiation may fail, in which case trainer is still None.
                if trainer is not None:
                    torch.cuda.synchronize()
                    for _ in range(batches_to_train):
                        inputs, targets = next(trainer._batch_generator(split="train"))
                        trainer._train_one_batch(batch_idx=0, inputs=inputs, targets=targets)
                    batch_size = mid
                    if trainer.is_chief:
                        logging.info(80 * "$")
                        logging.info(f"Batch size {batch_size} successful.")
                        logging.info(80 * "$")
                    lo = mid + 1
                    if not fails:
                        hi = hi * 2
            except RuntimeError as e:
                logging.warning(f'RuntimeError: "{e}"')
                if is_chief:
                    logging.info(80 * "-")
                    logging.info(f"Batch size {mid} FAILED.")
                    logging.info(80 * "-")
                if mid == 0 or (mid == 1 and fails):
                    # Cannot process any batches without erroring.
                    return 0
                fails += 1
                hi = mid
            finally:
                logging.info(torch.cuda.memory_allocated())
                del trainer
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logging.info(torch.cuda.memory_allocated())
                if fails > max_fails or lo == hi:
                    return batch_size
