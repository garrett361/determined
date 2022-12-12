import argparse
import gc
import json
import logging
import os
import pathlib
import random
import shutil
from typing import Callable, Generator, List, Literal, Tuple, Union

import data
import deepspeed
import determined as det
import numpy as np
import torch
import torch.nn as nn
import utils
from constants import DS_CONFIG_PATH, FLOPS_PROFILER_OUTPUT_PATH
from determined.pytorch import TorchData


class DeepSpeedTrainer(nn.Module):
    def __init__(
        self,
        core_context: det.core.Context,
        latest_checkpoint: str,
        args: argparse.Namespace,
        model: nn.Module,
        transforms: Union[Callable, List[Callable]],
        dataset_name: str,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.core_context = core_context
        self.latest_checkpoint = latest_checkpoint
        self.args = args
        self.model = model
        self.transforms = transforms
        self.dataset_name = dataset_name
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

    def _train_one_batch(self, inputs: TorchData, targets: TorchData) -> None:
        outputs = self.model_engine(inputs)
        loss = self.criterion(outputs, targets)
        self.model_engine.backward(loss)
        self.model_engine.step()

    def train_for_step(self) -> None:
        """Train for one SGD step, accounting for GAS."""
        for inputs, targets in self._batch_generator(split="train"):
            self._train_one_batch(inputs=inputs, targets=targets)
            if self.model_engine.is_gradient_accumulation_boundary():
                break

    def train_on_cluster(self) -> None:
        # A single op of fixed length is emitted.
        for op in self.core_context.searcher.operations():
            while self.steps_completed < op.length:
                self.train_for_step()
                self.steps_completed += 1
                if self.core_context.preempt.should_preempt():
                    return
            if self.is_chief:
                # Report completed value is not needed.
                op.report_completed(0)

    def autotuning(self) -> None:
        # A single op of fixed length is emitted.
        for op in self.core_context.searcher.operations():
            while self.steps_completed < op.length:
                self.train_for_step()
                self.steps_completed += 1
                if self.core_context.preempt.should_preempt():
                    return
            if self.is_chief:
                # Report completed value is not needed.
                op.report_completed(0)
        logging.warning("Saving autotuning results.")
        if self.is_chief:
            self._report_and_save_native_autotuning_results()

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
        cls, max_fails: int = 3, initial_batch_size: int = 1, **kwargs
    ) -> int:
        core_context = kwargs["core_context"]
        is_chief = core_context.distributed.rank == 0
        for op in core_context.searcher.operations():
            lo, hi = 1, 2 * initial_batch_size
            fails = 0
            batch_size = 0
            while fails <= max_fails:
                mid = (lo + hi) // 2
                trainer = None
                try:
                    if is_chief:
                        cls.update_tmbspg_in_config(train_micro_batch_size_per_gpu=mid)
                    kwargs["args"].train_micro_batch_size_per_gpu = mid
                    core_context.distributed.broadcast(None)  # Hack for syncing.
                    trainer = cls(**kwargs)
                    # Instantiation may fail, in which case trainer is still None.
                    if trainer is not None:
                        torch.cuda.synchronize()
                        for _ in range(op.length):
                            trainer.train_for_step()
                        batch_size = mid
                        if is_chief:
                            logging.info(80 * "$")
                            logging.warning(f"Batch size {batch_size} successful.")
                            logging.info(80 * "$")
                        lo = mid + 1
                        if not fails:
                            hi = hi * 2
                except RuntimeError as e:
                    logging.warning(f'RuntimeError: "{e}"')
                    if is_chief:
                        logging.info(80 * "-")
                        logging.warning(f"Batch size {mid} FAILED.")
                        logging.info(80 * "-")
                    if mid == 0 or (mid == 1 and fails):
                        # Cannot process any batches without erroring.
                        return 0
                    fails += 1
                    hi = mid
                finally:
                    # Memory cleanup.
                    del trainer
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if is_chief:
                        cls._report_and_save_flops_profiler_results(
                            batch_size=batch_size, core_context=core_context
                        )
                    if fails > max_fails or lo == hi:
                        if is_chief:
                            op.report_completed(batch_size)
                        return batch_size

    @staticmethod
    def _report_and_save_flops_profiler_results(
        batch_size: int,
        core_context: det.core.Context,
        src: Union[str, pathlib.Path] = FLOPS_PROFILER_OUTPUT_PATH,
    ) -> None:
        if os.path.exists(src):
            ds_profiler_results = utils.DSProfilerResults(path=src)
            metrics = ds_profiler_results.get_results_dict_from_path()
            core_context.train.report_validation_metrics(
                steps_completed=batch_size,
                metrics=metrics,
            )
            checkpoint_metadata_dict = {
                "steps_completed": batch_size,
                "micro_batch_size_per_gpu": batch_size,
            }
            with core_context.checkpoint.store_path(checkpoint_metadata_dict) as (
                path,
                storage_id,
            ):
                src = pathlib.Path(FLOPS_PROFILER_OUTPUT_PATH)
                dst = pathlib.Path(path).joinpath(src.name)
                shutil.copy(src=src, dst=dst)
            os.remove(src)

    def _report_and_save_native_autotuning_results(
        self, path: pathlib.Path = pathlib.Path(".")
    ) -> None:
        results = utils.DSAutotuningResults(path=path)
        ranked_results_dicts = results.get_ranked_results_dicts()
        for rank, results_dict in enumerate(ranked_results_dicts):
            metrics = results_dict["metrics"]
            ds_config = results_dict["exp_config"]["ds_config"]
            reported_metrics = utils.get_flattened_dict({**metrics, **ds_config})
            self.core_context.train.report_validation_metrics(
                steps_completed=rank,
                metrics=reported_metrics,
            )

        checkpoint_metadata_dict = {"steps_completed": len(ranked_results_dicts) - 1}
        with self.core_context.checkpoint.store_path(checkpoint_metadata_dict) as (
            ckpt_path,
            storage_id,
        ):
            for autotuning_dir in ("autotuning_exps", "autotuning_results"):
                src_path = pathlib.Path(autotuning_dir)
                shutil.copytree(
                    src=src_path,
                    dst=pathlib.Path(ckpt_path).joinpath(autotuning_dir),
                )
