import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import determined as det
import transformers
from determined.pytorch.deepspeed import dsat, overwrite_deepspeed_config
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

logging.basicConfig(level=logging.INFO)


class DetCallback(TrainerCallback):
    def __init__(
        self,
        core_context: det.core.Context,
        args: TrainingArguments,
        filter_metrics: List[str] = None,
        user_data: Dict = None,
    ) -> None:
        super().__init__()

        self.core_context = core_context

        self.filter_metrics = filter_metrics
        self.user_data = user_data
        self.load_last_checkpoint(args)

        self.last_metrics: Dict[str, float] = {"train_step": -1, "eval_step": -1}

        searcher_config = det.get_cluster_info().trial._config["searcher"]
        self.searcher_metric = searcher_config["metric"]
        self.searcher_unit = list(
            searcher_config.get("max_length", {"arbitrary": "batches"}).keys()
        )[0]
        self.searcher_max_length = list(
            searcher_config.get("max_length", {"arbitrary": 100}).values()
        )[0]
        self.searcher_ops = self.core_context.searcher.operations()
        self.current_op = next(self.searcher_ops)
        # self._check_searcher_compatibility(args)
        self.updating_searcher = False

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        metrics, metric_type = self._get_metrics(logs)
        if metric_type == TRAIN:
            # Prevents reporting metrics for the same step twice. This happens after
            # training is completed and average training metrics are reported with
            # the same step as the in-progress training metrics.
            if self.last_metrics["train_step"] != state.global_step:
                if state.is_world_process_zero:
                    self.core_context.train.report_training_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                metrics["train_step"] = state.global_step

        elif metric_type == EVAL:
            # Prevents reporting metrics for the same step twice. This happens when
            # after-training evaluation is completed, and it is reported with the same
            # step as the last during-training evaluation.
            if self.last_metrics["eval_step"] != state.global_step:
                if state.is_world_process_zero:
                    self.core_context.train.report_validation_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                metrics["eval_step"] = state.global_step
        else:
            logging.warning(f"Metrics not reported: metric type = {metric_type}.")

        self.last_metrics.update(metrics)

        # Update searcher state after collecting the metrics.
        if self.updating_searcher is True:
            self._update_searcher(state, control)

        # If searcher is NOT being updated and preemption signal is received
        # (e.g., by pausing experiment in the WebUI), notify Trainer (via TrainerControl)
        # to save the checkpoint. After the checkpoint is uploaded to Determined storage,
        # the process is preempted (see on_save() method for details).
        if self.updating_searcher is False and self.core_context.preempt.should_preempt():
            control.should_save = True

    def _get_metrics(self, logs: Dict) -> Tuple[Dict, str]:
        metrics = logs
        metric_type = get_metric_type(logs)
        if self.filter_metrics:
            metrics = {}
            for k, v in logs.items():
                if any(m in k for m in self.filter_metrics) is True:
                    metrics[k] = v

        return metrics, metric_type

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        info = det.get_cluster_info()
        assert info

        # local_path is where HF Trainer saves model and tokenizer in a given step.
        local_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if state.is_world_process_zero:
            if self.user_data is not None:
                self._on_save_user_data(local_path)

        det_checkpoint_metadata = {
            "steps_completed": state.global_step,
            "trial_id": info.trial.trial_id,
        }

        def selector(x: str) -> bool:
            return x.startswith((f"checkpoint-{state.global_step}/", "runs/"))

        self.core_context.checkpoint.upload(
            args.output_dir, metadata=det_checkpoint_metadata, shard=True, selector=selector
        )

        if self.core_context.preempt.should_preempt():
            raise Exception("Process preempted / killed")

    def _on_save_user_data(self, save_path: str) -> None:
        """
        User-defined saving of objects from self.checkpoint_metadata under save_path.
        After objects are saved, Determined handles uploading and downloading objects to/from selected storage.
        """
        with open(os.path.join(save_path, "my_data.json"), "w") as f:
            json.dump(self.user_data, f)

    def load_last_checkpoint(self, args: TrainingArguments) -> None:
        info = det.get_cluster_info()
        assert info

        latest_checkpoint = info.latest_checkpoint
        if latest_checkpoint is not None:
            if args.overwrite_output_dir is True:
                logging.info(
                    f"Skip downloading last checkpoint from Determined due "
                    f"to overwrite_output_dir=True."
                )
                return

            # To resume DeepSpeed, each node requires ALL sharded model/optimizer states,
            # so we can skip using selector and just download all files.
            self.core_context.checkpoint.download(latest_checkpoint, args.output_dir)

            checkpoint_path = get_last_checkpoint(args.output_dir)
            args.resume_from_checkpoint = checkpoint_path

            logging.info(f"Latest checkpoint downloaded to {checkpoint_path}.")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # state.epoch is not None only during training.
        if state.epoch and self.searcher_unit == "batches":
            if state.is_world_process_zero:
                self.current_op.report_progress(state.global_step)

            if state.global_step >= self.current_op.length:
                logging.info(
                    f"Max length of {self.current_op.length} steps reached for current "
                    f"searcher operation. Updating searcher."
                )
                self._update_searcher(state, control)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # state.epoch is not None only during training.
        if state.epoch and self.searcher_unit == "epochs":
            if state.is_world_process_zero:
                self.current_op.report_progress(state.epoch)

            if state.epoch >= self.current_op.length:
                logging.info(
                    f"Max length of {state.epoch} epochs reached for current "
                    f"searcher operation. Updating searcher."
                )
                self._update_searcher(state, control)

    def _update_searcher(self, state: TrainerState, control: TrainerControl) -> None:
        if self._metrics_reported(state.global_step) is False:
            self._wait_for_metrics(control)
            return

        if state.is_world_process_zero:
            if self.last_metrics is None:
                logging.warning(
                    f"No training or evaluation metrics has been recorded. Please check your settings for "
                    f"training metrics (--logging_strategy and --logging_steps) or "
                    f"evaluation metrics (--evaluation_strategy and --eval_steps). "
                    f"Reporting trainer_state.best_metric to the searcher."
                )
                searcher_metric = state.best_metric
            elif self.searcher_metric not in self.last_metrics:
                logging.warning(
                    f"Searcher metric {self.searcher_metric} from the yaml config file does not match any "
                    f"of the recorded metrics in {self.last_metrics}. "
                    f"Reporting trainer_state.best_metric to the searcher."
                )
                searcher_metric = state.best_metric
            else:
                searcher_metric = self.last_metrics[self.searcher_metric]

            logging.info(f"Metric reported to searcher: {searcher_metric}")
            self.current_op.report_completed(searcher_metric)

        self.updating_searcher = False

        try:
            self.current_op = next(self.searcher_ops)
        except StopIteration:
            control.should_training_stop = True

    def _metrics_reported(self, step: int) -> bool:
        return self.last_metrics["eval_step"] == step and self.last_metrics["train_step"] == step

    def _wait_for_metrics(self, control: TrainerControl) -> None:
        # Notify Trainer (via TrainerControl) to:
        # (1) log current training metrics,
        # (2) evaluate the model and log evaluation metrics,
        # (3) save the checkpoint.
        #  updating_searcher is as an internal flag that indicates we are
        #  in the process of updating the searcher with the current metrics.
        control.should_log = True
        control.should_evaluate = True
        control.should_save = True
        self.updating_searcher = True

    def _check_searcher_compatibility(self, args: TrainingArguments) -> None:
        if self.searcher_unit == "batches":
            if args.max_steps == -1:
                self._log_config_mismatch("epochs", args.num_train_epochs)
            elif args.max_steps != self.searcher_max_length:
                self._log_config_mismatch("batches", args.max_steps)
        elif self.searcher_unit == "epochs":
            if args.max_steps != -1:
                self._log_config_mismatch("batches", args.max_steps)
            elif args.num_train_epochs != self.searcher_max_length:
                self._log_config_mismatch("epochs", args.num_train_epochs)

    def _log_config_mismatch(
        self,
        trainer_units: str,
        trainer_len: float,
    ) -> None:
        logging.warning(
            f"Searcher configuration does not match HF Trainer configuration. "
            f"Searcher uses {self.searcher_unit}={self.searcher_max_length}, "
            f"while HF Trainer uses {trainer_units}={trainer_len}. "
            f"Continuing this run may cause Searcher not to behave correctly. "
            f"Make sure to match the units between HF Trainer and Searcher: "
            f"use (--num_train_epochs and searcher.max_length.epochs) OR "
            f"(--max_steps and searcher.max_length.batches)."
        )


EVAL = "eval_"
TEST = "test_"
TRAIN_AVG = "train_"
TRAIN = "train_progress"


def get_metric_type(d):
    for k, v in d.items():
        if k.startswith(EVAL):
            return EVAL
        elif k.startswith(TEST):
            return TEST
        elif k.startswith(TRAIN_AVG):
            return TRAIN_AVG
        else:
            return TRAIN


def get_ds_config_path_from_args(args: List[str]) -> Optional[str]:
    for idx in range(len(args)):
        if args[idx] == "--deepspeed":
            ds_config_idx = idx + 1
            ds_config_path = args[ds_config_idx]
            return ds_config_path


def replace_ds_config_file_using_overwrites(
    args: List[str], hparams: Dict[str, Any], overwrite_key: str = dsat._defaults.OVERWRITE_KEY
):
    """
    Gets the deepspeed json config path from the list of HF args, overwrites its values using
    the provided overwrite values, and the re-writes the result to the original config path.
    """
    ds_config_path = get_ds_config_path_from_args(args)
    with open(ds_config_path, "r") as f:
        ds_config_dict_with_overwrites = json.load(f)
        # If overwrites are provided, use them. The deepspeed configuration is assumed to have a
        # consistent batch size configuration at this point, with all of train_batch_size,
        # train_micro_batch_size_per_gpu, and gradient_accumulation_steps filled in.
        ds_config_dict_with_overwrites = overwrite_deepspeed_config(
            ds_config_dict_with_overwrites, hparams.get(overwrite_key, {})
        )
        print("writing ds config to file", ds_config_dict_with_overwrites)
        # overwrite the original config
        with open(ds_config_path, "w") as f:
            json.dump(ds_config_dict_with_overwrites, f)


def create_consistent_hf_args_for_deepspeed(args: List[str], ds_config_path: str) -> List[str]:
    """
    TODO: Cleanup and write actual doc string. Remove print tests.

    Helper function which modifies the *HFConfig* batch-size-related args based on the deepspeed
    json file, if present.

    The default HF behavior is to use "auto" for these values in the json file when combining
    deepspeed and HF Trainer, and then HF will populate the ds config values based on other
    HF args. This helper function exactly reverses the logic and modifies the HF args based on the
    ds json config values, if they are not "auto".
    """
    # Next we need to ensure that the HF batch config aligns with the deepspeed one.
    print("args before", args)
    with open(ds_config_path, "r") as f:
        ds_config_dict = json.load(f)

    hf_flag_to_ds_key_dict = {
        "--per_device_train_batch_size": "train_micro_batch_size_per_gpu",
        "--gradient_accumulation_steps": "gradient_accumulation_steps",
    }
    seen_hf_flags = set()

    for idx in range(len(args)):
        for hf_flag, ds_key in hf_flag_to_ds_key_dict.items():
            if args[idx] == hf_flag:
                overwrite_value = str(ds_config_dict[ds_key])
                if overwrite_value != "auto" and args[idx + 1] != overwrite_value:
                    logging.warning(
                        f"Changing {hf_flag} from {args[idx +1]} to {overwrite_value} to match the DeepSpeed configuration."
                    )
                    args[idx + 1] = overwrite_value
                seen_hf_flags.add(hf_flag)

    # Add any unseen flags in, so that we don't just fall back to HF defaults.
    for hf_flag, ds_key in hf_flag_to_ds_key_dict.items():
        if hf_flag not in seen_hf_flags:
            args.extend([hf_flag, str(ds_config_dict[ds_key])])
    print("args after", args)

    return args
