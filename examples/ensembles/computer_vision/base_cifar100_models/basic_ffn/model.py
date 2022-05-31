"""
CNN on Cifar10 from Keras example:
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
"""
from typing import Any, Dict, Sequence, Union, Tuple

import attrdict
import torch
from torch import nn

import data

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class Model(nn.Module):
    def __init__(self, dataset_metadata: Dict, hidden_layers: Tuple[int]):
        super(Model, self).__init__()
        self.dataset_metadata = attrdict.AttrDict(dataset_metadata)
        self.hidden_layers = hidden_layers

        input_shape = (
            self.dataset_metadata.img_size ** 2 * self.dataset_metadata.in_chans
        )
        linear_layers = []
        in_dims = [input_shape] + self.hidden_layers[:-1]
        out_dims = self.hidden_layers
        for (in_dim, out_dim) in zip(in_dims, out_dims):
            linear_layers.append(nn.Linear(in_dim, out_dim))
        self.linear_layers = nn.ModuleList(linear_layers)
        self.class_layer = nn.Linear(
            self.hidden_layers[-1], self.dataset_metadata.num_classes
        )

    def _build_inference_transform(self) -> nn.Module:
        return data.build_transform(self.dataset_metadata, {}, True)

    def forward(self, input) -> torch.Tensor:
        output = input.flatten(start_dim=1)
        for layer in self.linear_layers:
            output = layer(output)
            output = output.relu()
        output = self.class_layer(output)
        return output

    def inference(self, image) -> torch.Tensor:
        # Expects PIL image
        pass
        pass
