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
        return data.build_transform(self.dataset_metadata)

    def forward(self, input) -> torch.Tensor:
        output = input.flatten(start_dim=1)
        for layer in self.linear_layers:
            output = layer(output)
            output = output.relu()
        output = self.class_layer(output)
        return output

    def inference(self, image) -> torch.Tensor:
        # Takes in a single PIL image and outputs class probabilities.
        self.eval()
        with torch.no_grad():
            trans = self._build_inference_transform()
            image_t = trans(image)[None]  # Restore batch dimension.
            logits = self(image_t)
            prob = logits.softmax(dim=-1).flatten()
            prob_list = [
                (p.item(), label)
                for p, label in zip(prob, self.dataset_metadata.labels)
            ]
            return sorted(prob_list, reverse=True)
