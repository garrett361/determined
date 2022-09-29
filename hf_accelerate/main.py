import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

dimension = batch_size = 10


class RandomDataset(Dataset):
    def __init__(self, dimension: int = dimension, num_records: int = 100) -> None:
        self.dimension = dimension
        self.num_records = num_records
        self.records = torch.randn(num_records, dimension)

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.records[idx]


if __name__ == "__main__":
    model = nn.Linear(dimension, dimension)
    dataloader = DataLoader(RandomDataset(), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for batch_idx, batch in enumerate(dataloader):
        output = model(batch)
        loss = F.mse_loss(output, batch)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss at batch_idx {batch_idx}: {loss.item()}")
