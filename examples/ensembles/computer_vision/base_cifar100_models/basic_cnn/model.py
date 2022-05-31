"""
CNN on Cifar10 from Keras example:
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
"""
from typing import Any, Dict, Sequence, Union

import attrdict
import torch
from torch import nn

import data

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class Flatten(nn.Module):
    def forward(self, *args: TorchData, **kwargs: Any) -> torch.Tensor:
        assert len(args) == 1
        x = args[0]
        assert isinstance(x, torch.Tensor)
        return x.contiguous().view(x.size(0), -1)


class Model(nn.Module):
    def __init__(
        self,
        dataset_metadata: Dict,
        layer1_dropout: float,
        layer2_dropout: float,
        layer3_dropout: float,
    ):
        super(Model, self).__init__()
        self.dataset_metadata = attrdict.AttrDict(dataset_metadata)
        self.model = nn.Sequential(
            nn.Conv2d(
                self.dataset_metadata.in_chans,
                self.dataset_metadata.img_size,
                kernel_size=(3, 3),
            ),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(layer1_dropout),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(layer2_dropout),
            Flatten(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout2d(layer3_dropout),
            nn.Linear(512, self.dataset_metadata.num_classes),
        )

    def _build_inference_transform(self) -> nn.Module:
        return data.build_transform(self.dataset_metadata, {}, True)

    def forward(self, input) -> torch.Tensor:
        return self.model(input)
