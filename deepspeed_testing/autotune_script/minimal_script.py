import json

import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import Dataset


DIM = 2 ** 6
NUM_RECORDS = 10 ** 4

with open("ds_config.json") as f:
    ds_config = json.load(f)


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
            self.linear = nn.Linear(DIM, DIM)

        def forward(self, x):
            return self.linear(x)

    net = Net()

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        model=net, model_parameters=net.parameters(), training_data=trainset, config=ds_config
    )

    fp16 = model_engine.fp16_enabled()

    criterion = nn.MSELoss()

    for batch in trainloader:
        # get the inputs; batch is a list of [inputs, labels]
        inputs, labels = batch[0].to(model_engine.local_rank), batch[1].to(model_engine.local_rank)
        if fp16:
            inputs = inputs.half()
            labels = labels.half()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()


if __name__ == "__main__":
    main()
