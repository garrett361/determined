import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from determined import pytorch


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len: int) -> None:
        self.dataset_len = dataset_len

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, index: int) -> Tuple:
        return torch.Tensor([float(1)])


class NoopPyTorchTrial(pytorch.PyTorchTrial):
    def __init__(self, context: pytorch.PyTorchTrialContext):
        self.context = context
        self.dataset_len = context.get_hparam("dataset_len")

        model = nn.Linear(1, 1, False)
        model.weight.data.fill_(0)

        self.model = context.wrap_model(model)

        opt = torch.optim.SGD(self.model.parameters(), 0.1)
        self.opt = context.wrap_optimizer(opt)

    def train_batch(
        self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        w_real = self.model.weight.data[0]

        loss = torch.nn.MSELoss()(self.model(batch), batch)

        self.context.backward(loss)
        self.context.step_optimizer(self.opt)

        rank = self.context.distributed.get_rank()
        print(f"finished train_batch for rank {rank}")
        print(f"rank {rank} finished batch {batch_idx} in epoch {epoch_idx}")

        return {"loss": loss, "w_real": w_real}

    def evaluate_batch(self, batch: pytorch.TorchData) -> Dict[str, Any]:
        val = batch[0]
        np_rand = np.random.randint(1, 1000)
        rand_rand = random.randint(0, 1000)
        torch_rand = torch.randint(1000, (1,))
        gpu_rand = torch.randint(1000, (1,), device=self.context.device)

        print(f"finished evaluate_batch for rank {self.context.distributed.get_rank()}")

        return {
            "validation_error": val,
            "np_rand": np_rand,
            "rand_rand": rand_rand,
            "torch_rand": torch_rand,
            "gpu_rand": gpu_rand,
        }

    def build_training_data_loader(self):
        return pytorch.DataLoader(
            OnesDataset(self.dataset_len), batch_size=self.context.get_per_slot_batch_size()
        )

    def build_validation_data_loader(self):
        return pytorch.DataLoader(
            OnesDataset(self.dataset_len), batch_size=self.context.get_per_slot_batch_size()
        )


class NoopPyTorchTrialWithCallbacks(NoopPyTorchTrial):
    def __init__(self, context: pytorch.PyTorchTrialContext):
        super().__init__(context)

    def build_callbacks(self) -> Dict[str, pytorch.PyTorchCallback]:
        return {"test_callbacks": TestCallbacks()}

    def train_batch(
        self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        rank = loss = self.context.distributed.get_rank()
        print(f"finished train_batch for rank {rank}, loss = {loss}")
        return {"loss": loss}


class TestCallbacks(pytorch.PyTorchCallback):
    def __init__(self) -> None:
        pass

    def on_training_workload_end(
        self, avg_metrics: List[Dict[str, Any]], batch_metrics: Dict[str, Any]
    ) -> None:
        print("Calling on_training_workload_end")
        print("avg_metrics", avg_metrics)
        print("batch_metrics", batch_metrics)

    def on_checkpoint_upload_end(self, uuid: str) -> None:
        print(f"Calling on_checkpoint_upload_end. uuid={uuid}")

    def on_checkpoint_end(self, checkpoint_dir: str) -> None:
        print("Deprecated on_checkpoint_end call")
