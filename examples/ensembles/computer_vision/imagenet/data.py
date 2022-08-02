import dataclasses
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
import os

import attrdict
import pandas as pd
import pickle
from timm.data import create_transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

ImageStat = Union[Tuple[float], Tuple[float, float, float]]
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

# Path to the train/val/test index splitting pkl file.
SPLIT_PICKLE_PATH = "imagenetv2_train_val_test_idx_mappings.pkl"


class RAMImageFolder:
    """Loads the usual ImageFolder results into memory. Can accept a list or tuple of transforms."""

    def __init__(
        self,
        root: str,
        target_transform: Callable,
        transforms: Union[Callable, Sequence[Callable]],
    ) -> None:
        if not isinstance(transforms, Sequence):
            transforms = [transforms]
        im_folder = ImageFolder(root=root, target_transform=target_transform)
        self.samples = []
        for im, target in im_folder:
            sample = [transform(im) for transform in transforms]
            sample.append(target)
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> TorchData:
        return self.samples[idx]


# ImageNetv2 labels its directories with the corresponding ImageNet idx. This requires special
# handling to map back to the appropriate idxs.
class SplitImageNetv2ImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        transforms: Union[Callable, Sequence[Callable]],
        split_pkl_path: str = SPLIT_PICKLE_PATH,
    ) -> None:
        super().__init__(root=root)
        with open(split_pkl_path, "rb") as f:
            self._split_mappings = pickle.load(f)
        self.idx_mapping = self._split_mappings[split]
        if not isinstance(transforms, Sequence):
            transforms = [transforms]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.idx_mapping)

    def __getitem__(self, idx: int) -> TorchData:
        index = self.idx_mapping[idx]
        path, target = self.samples[index]
        img = self.loader(path)
        transformed_imgs = [transform(img) for transform in self.transforms]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return transformed_imgs, target

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
        return classes, class_to_idx


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
    "imagenetv2-matched-frequency": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/data/imagenetv2-matched-frequency-format-val",
        dataset_class=SplitImageNetv2ImageFolder,
    ),
    "imagenetv2-threshold0.7": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/data/imagenetv2-threshold0.7-format-val",
        dataset_class=SplitImageNetv2ImageFolder,
    ),
    "imagenetv2-top-images": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/data/imagenetv2-top-images-format-val",
        dataset_class=SplitImageNetv2ImageFolder,
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


def build_data_transform(dataset_metadata: attrdict.AttrDict, model_data: pd.Series) -> Callable:
    """Generate transforms via timm's transform factory."""
    return create_transform(
        input_size=model_data.img_size,
        is_training=False,
        mean=dataset_metadata.mean,
        std=dataset_metadata.std,
        interpolation=model_data.interpolation,
        crop_pct=model_data.crop_pct,
    )


def get_dataset(
    name: str,
    split: Literal["train", "val", "test"],
    transforms: Union[Callable, List[Callable]] = None,
) -> Dataset:
    dataset_metadata = DATASET_METADATA_BY_NAME[name].to_attrdict()
    if dataset_metadata.dataset_class == RAMImageFolder:
        root = dataset_metadata.root + "/" + split
        target_transform = build_target_transform(dataset_metadata.target_transform_path)
        dataset = dataset_metadata.dataset_class(
            root=root, transforms=transforms, target_transform=target_transform
        )
    elif dataset_metadata.dataset_class == SplitImageNetv2ImageFolder:
        dataset = dataset_metadata.dataset_class(
            split=split,
            root=dataset_metadata.root,
            transforms=transforms,
        )
    return dataset
