from typing import Any, Dict, Sequence, Union, Tuple

import attrdict
import torch
from torch import nn
import torchvision

import data

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class Model(nn.Module):
    def __init__(self, dataset_metadata: Dict, pretrained: bool):
        super(Model, self).__init__()
        self.dataset_metadata = attrdict.AttrDict(dataset_metadata)
        self.model = torchvision.models.resnet50(pretrained=pretrained)

    def _build_inference_transform(self) -> nn.Module:
        return data.build_transform(self.dataset_metadata)

    def forward(self, input) -> torch.Tensor:
        return self.model(input)

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
