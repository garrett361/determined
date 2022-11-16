import argparse

import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import Dataset


DIM = 2 ** 6
LAYERS = 16
NUM_RECORDS = 10 ** 4


def parse_args():
    parser = argparse.ArgumentParser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    # Absorb a possible `local_rank` arg from the launcher.
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local rank passed from distributed launcher"
    )

    args = parser.parse_args()

    return args


def main():
    # GG: Why is this needed?
    deepspeed.init_distributed()

    class RandomDataset(Dataset):
        def __init__(self):
            self.inputs = torch.randn(NUM_RECORDS, DIM)
            self.targets = torch.randn(NUM_RECORDS, DIM)

        def __len__(self):
            return NUM_RECORDS

        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    trainset = RandomDataset()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layers = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(LAYERS)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    net = Net()
    args = parse_args()
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=net.parameters(),
        training_data=trainset,
    )

    fp16 = model_engine.fp16_enabled()

    criterion = nn.MSELoss()

    for batch in trainloader:
        # get the inputs; batch is a list of [inputs, labels]
        inputs, labels = batch[0].to(model_engine.local_rank), batch[1].to(model_engine.local_rank)
        if fp16:
            inputs = inputs.half()
            labels = labels.half()
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()


if __name__ == "__main__":
    main()
