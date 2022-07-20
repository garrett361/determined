import dataclasses
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import attrdict
import pickle
from timm.data import create_transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

ImageStat = Union[Tuple[float], Tuple[float, float, float]]
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

# We only use timm models which have the following:
INTERPOLATION = "bicubic"
CROP_PCT = 0.875
# Path to the train/val/test index splitting pkl file.
SPLIT_PICKLE_PATH = "imagenetv2_train_val_test_idx_mappings.pkl"


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
        self,
        split: Literal["train", "val", "test"],
        split_pkl_path: str = SPLIT_PICKLE_PATH,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        with open(split_pkl_path, "rb") as f:
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


@dataclasses.dataclass
class DatasetMetadata:
    num_classes: int
    root: str
    dataset_class: type = ImageFolder
    img_size: int = 224
    in_chans: int = 3
    mean: ImageStat = (0.485, 0.456, 0.406)
    std: ImageStat = (0.229, 0.224, 0.225)
    target_transform_path: Optional[str] = None

    def to_attrdict(self) -> Dict[str, Union[int, ImageStat]]:
        return attrdict.AttrDict(dataclasses.asdict(self))


DATASET_METADATA_BY_NAME = {
    "imagenet": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/data/imagenet",
        dataset_class=SplitImageFolder,
    ),
    "imagenetv2-matched-frequency": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/data/imagenetv2-matched-frequency-format-val",
        dataset_class=SplitImageFolder,
    ),
    "imagenetv2-threshold0.7": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/data/imagenetv2-threshold0.7-format-val",
        dataset_class=SplitImageFolder,
    ),
    "imagenetv2-top-images": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/data/imagenetv2-top-images-format-val",
        dataset_class=SplitImageFolder,
    ),
    "imagewang": DatasetMetadata(
        num_classes=20,
        root="shared_fs/data/imagewang",
        dataset_class=RAMImageFolder,
        target_transform_path="imagewang_to_imagenet_idx_map.pkl",
    ),
    "imagewang-160": DatasetMetadata(
        num_classes=20,
        root="shared_fs/data/imagewang-160",
        dataset_class=RAMImageFolder,
        target_transform_path="imagewang_to_imagenet_idx_map.pkl",
    ),
    "imagewang-320": DatasetMetadata(
        num_classes=20,
        root="shared_fs/data/imagewang-320",
        dataset_class=RAMImageFolder,
        target_transform_path="imagewang_to_imagenet_idx_map.pkl",
    ),
    "imagewoof2": DatasetMetadata(
        num_classes=10,
        root="shared_fs/data/imagewoof2",
        dataset_class=RAMImageFolder,
        target_transform_path="imagewoof2_to_imagenet_idx_map.pkl",
    ),
    "imagewoof2-160": DatasetMetadata(
        num_classes=10,
        root="shared_fs/data/imagewoof2-160",
        dataset_class=RAMImageFolder,
        target_transform_path="imagewoof2_to_imagenet_idx_map.pkl",
    ),
    "imagewoof2-320": DatasetMetadata(
        num_classes=10,
        root="shared_fs/data/imagewoof2-320",
        dataset_class=RAMImageFolder,
        target_transform_path="imagewoof2_to_imagenet_idx_map.pkl",
    ),
    "imagenette2": DatasetMetadata(
        num_classes=10,
        root="shared_fs/data/imagenette2",
        dataset_class=RAMImageFolder,
        target_transform_path="imagenette2_to_imagenet_idx_map.pkl",
    ),
    "imagenette2-160": DatasetMetadata(
        num_classes=10,
        root="shared_fs/data/imagenette2-160",
        dataset_class=RAMImageFolder,
        target_transform_path="imagenette2_to_imagenet_idx_map.pkl",
    ),
    "imagenette2-320": DatasetMetadata(
        num_classes=10,
        root="shared_fs/data/imagenette2-320",
        dataset_class=RAMImageFolder,
        target_transform_path="imagenette2_to_imagenet_idx_map.pkl",
    ),
}


def build_target_transform(path: str) -> Callable:
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    label_transform = lambda idx: mapping[idx]
    return label_transform


def build_basic_train_transform(
    dataset_metadata: attrdict.AttrDict,
    transform_config: Optional[dict] = None,
) -> nn.Module:
    """Generate transforms via timm's transform factory, but always using training mode and bicubic
    interpolation and crop_pct = 0.875. We will also filter the base models on these criteria.
    """
    transform_config = transform_config or {}
    return create_transform(
        input_size=dataset_metadata.img_size,
        is_training=False,
        mean=dataset_metadata.mean,
        std=dataset_metadata.std,
        interpolation=INTERPOLATION,
        crop_pct=CROP_PCT,
        **transform_config,
    )


def get_dataset(
    name: str,
    split: Literal["train", "val", "test"],
    transform_config: Optional[dict] = None,
) -> Dataset:
    transform_config = transform_config or {}
    dataset_metadata = DATASET_METADATA_BY_NAME[name].to_attrdict()
    transform = build_basic_train_transform(dataset_metadata, transform_config=transform_config)
    if dataset_metadata.dataset_class == RAMImageFolder:
        root = dataset_metadata.root + "/" + split
        target_transform = build_target_transform(dataset_metadata.target_transform_path)
        dataset = dataset_metadata.dataset_class(
            root=root, transform=transform, target_transform=target_transform
        )
    elif dataset_metadata.dataset_class == SplitImageFolder:
        dataset = dataset_metadata.dataset_class(
            split=split,
            root=dataset_metadata.root,
            transform=transform,
        )
    return dataset
