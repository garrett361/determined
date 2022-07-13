import dataclasses
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import pickle
from timm.data import create_transform
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder

ImageStat = Union[Tuple[float], Tuple[float, float, float]]
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


@dataclasses.dataclass
class DatasetMetadata:
    num_classes: int
    img_size: int
    in_chans: int
    mean: ImageStat
    std: ImageStat

    def to_dict(self) -> Dict[str, Union[int, ImageStat]]:
        return dataclasses.asdict(self)


DATASET_METADATA_BY_NAME = {
    "mnist": DatasetMetadata(
        num_classes=10, img_size=28, in_chans=1, mean=(0.1307,), std=(0.3081,)
    ),
    "cifar10": DatasetMetadata(
        num_classes=10,
        img_size=32,
        in_chans=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "imagenet": DatasetMetadata(
        num_classes=1000,
        img_size=224,
        in_chans=3,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}


class RAMImageFolder:
    """Loads the usual ImageFolder results into memory."""

    def __init__(self, *args, **kwargs) -> None:
        im_folder = ImageFolder(*args, **kwargs)
        self.samples = []
        for im_t, label_t in im_folder:
            self.samples.append((im_t, label_t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> TorchData:
        return self.samples[idx]


class SplitImageFolder(ImageFolder):
    def __init__(
        self, split_path: str, split: Literal["train", "val", "test"], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        with open(split_path, "rb") as f:
            self._split_mappings = pickle.load(f)
        self.idx_mapping = self._split_mappings[split]

    def __len__(self) -> int:
        return len(self.idx_mapping)

    def __getitem__(self, idx: int) -> TorchData:
        index = self.idx_mapping[idx]
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def build_transform(
    dataset_metadata: Any, transform_config: Optional[dict] = None, train: bool = False
) -> nn.Module:
    """Generate transforms via timm's transform factory."""
    if transform_config is None:
        transform_config = {}
    return create_transform(
        input_size=dataset_metadata.img_size,
        is_training=train,
        mean=dataset_metadata.mean,
        std=dataset_metadata.std,
        **transform_config,
    )
