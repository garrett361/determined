import os

import torch
import torch.distributed as dist
import torch.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.transforms as T
from tqdm import tqdm


class MNISTModel(nn.Module):
    def __init__(
        self,
        n_filters1: int = 32,
        n_filters2: int = 64,
        dropout1: float = 0.25,
        dropout2: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, n_filters1, 3, 1),
            nn.ReLU(),
            nn.Conv2d(
                n_filters1,
                n_filters2,
                3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout1),
            nn.Flatten(),
            nn.Linear(144 * n_filters2, 128),
            nn.ReLU(),
            nn.Dropout2d(dropout2),
            nn.Linear(128, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def build_data_loader(is_distributed: bool, train: bool, batch_size: int = 256) -> DataLoader:
    mnist_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(root="/data", train=train, download=True, transform=mnist_transform)
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    return loader


def train(model, device, optimizer, criterion, train_loader, epochs=1) -> None:
    model.train()
    for epoch_idx in tqdm(range(epochs), desc="training_epoch"):
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="training_batch")):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print(f"Train Epoch: {epoch_idx} Batch: {batch_idx} Loss: {loss.item()}")


def validate(model, device, criterion, val_loader) -> None:
    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        print(f"DEVICE: {device} Batch: {batch_idx} Loss: {loss.item()}")
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch: {batch_idx} Loss: {loss.item()}")


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    local_rank = 0 if not is_distributed else int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    if is_distributed:
        dist.init_process_group("nccl", world_size=world_size, rank=local_rank)

    train_loader = build_data_loader(is_distributed, train=True)
    val_loader = build_data_loader(is_distributed, train=False)

    model = MNISTModel()
    model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        device=device,
        epochs=1,
    )

    validate(model=model, criterion=criterion, val_loader=val_loader, device=device)


if __name__ == "__main__":
    main()
