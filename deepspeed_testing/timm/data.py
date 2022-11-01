import dataclasses
import os
from inspect import signature
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union


import attrdict
import pandas as pd
import pickle
import timm
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

SMALL_TIMM_MODELS_DF = pd.read_feather("models/small_timm_models.feather").set_index("model")
TOP_TIMM_MODELS_DF = pd.read_feather("models/top_timm_models.feather").set_index("model")
ALL_MODELS_DF = pd.concat([SMALL_TIMM_MODELS_DF, TOP_TIMM_MODELS_DF])

ImageStat = Union[Tuple[float], Tuple[float, float, float]]
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


IMAGENETV2_SPLIT_PICKLE_PATH = "idx_maps/imagenetv2_train_val_test_idx_mappings.pkl"
DATA_SPLITS = ("train", "val", "test")
IMAGEWANG_IDX_MAP_DICT = {
    "train": "idx_maps/imagewang_train_to_imagenet_idx_map.pkl",
    "val": "idx_maps/imagewang_val_to_imagenet_idx_map.pkl",
    "test": None,
}
IMAGEWOOF2_IDX_MAP_DICT = {
    split: "idx_maps/imagewoof2_to_imagenet_idx_map.pkl" for split in DATA_SPLITS
}
IMAGENETTE2_IDX_MAP_DICT = {
    split: "idx_maps/imagenette2_to_imagenet_idx_map.pkl" for split in DATA_SPLITS
}


class MultiTransImageFolder:
    """The basic ImageFolder, but with multiple transforms applied to the images."""

    def __init__(
        self,
        root: str,
        target_transform: Callable,
        transforms: Optional[Union[Callable, Sequence[Callable]]] = None,
    ) -> None:
        self.transforms = transforms
        self.im_folder = ImageFolder(root=root, target_transform=target_transform)

    def __len__(self) -> int:
        return len(self.im_folder)

    def __getitem__(self, idx) -> Tuple[List[torch.Tensor], torch.Tensor]:
        samples, target = self.im_folder[idx]
        if isinstance(self.transforms, Sequence):
            samples = [t(samples) for t in self.transforms]
        elif self.transforms is not None:
            samples = self.transforms(samples)
        return samples, target


# ImageNetv2 labels its directories with the corresponding ImageNet idx. This requires special
# handling to map back to the appropriate idxs.
class SplitImageNetv2ImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        transforms: Optional[Union[Callable, Sequence[Callable]]] = None,
        split_pkl_path: str = IMAGENETV2_SPLIT_PICKLE_PATH,
    ) -> None:
        super().__init__(root=root)
        with open(split_pkl_path, "rb") as f:
            self._split_mappings = pickle.load(f)
        self.idx_mapping = self._split_mappings[split]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.idx_mapping)

    def __getitem__(self, idx: int) -> TorchData:
        index = self.idx_mapping[idx]
        path, target = self.samples[index]
        samples = self.loader(path)
        if isinstance(self.transforms, Sequence):
            samples = [t(samples) for t in self.transforms]
        elif self.transforms is not None:
            samples = self.transforms(samples)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return samples, target

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
    target_transform_paths: Optional[Dict] = dataclasses.field(
        default_factory=lambda: {split: None for split in DATA_SPLITS}
    )

    def to_attrdict(self) -> Dict[str, Union[int, ImageStat]]:
        return attrdict.AttrDict(dataclasses.asdict(self))


DATASET_METADATA_BY_NAME = {
    "mini_imagenet": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/ensembles/data/mini_imagenet",
        dataset_class=MultiTransImageFolder,
    ),
    "imagenet": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/ensembles/data/imagenet",
        dataset_class=MultiTransImageFolder,
    ),
    "imagenetv2-matched-frequency": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/ensembles/data/imagenetv2-matched-frequency-format-val",
        dataset_class=SplitImageNetv2ImageFolder,
    ),
    "imagenetv2-threshold0.7": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/ensembles/data/imagenetv2-threshold0.7-format-val",
        dataset_class=SplitImageNetv2ImageFolder,
    ),
    "imagenetv2-top-images": DatasetMetadata(
        num_classes=1000,
        root="shared_fs/ensembles/data/imagenetv2-top-images-format-val",
        dataset_class=SplitImageNetv2ImageFolder,
    ),
    "imagewang": DatasetMetadata(
        num_classes=20,
        root="shared_fs/ensembles/data/imagewang",
        dataset_class=MultiTransImageFolder,
        target_transform_paths=IMAGEWANG_IDX_MAP_DICT,
    ),
    "imagewang-160": DatasetMetadata(
        num_classes=20,
        root="shared_fs/ensembles/data/imagewang-160",
        dataset_class=MultiTransImageFolder,
        target_transform_paths=IMAGEWANG_IDX_MAP_DICT,
    ),
    "imagewang-320": DatasetMetadata(
        num_classes=20,
        root="shared_fs/ensembles/data/imagewang-320",
        dataset_class=MultiTransImageFolder,
        target_transform_paths=IMAGEWANG_IDX_MAP_DICT,
    ),
    "imagewoof2": DatasetMetadata(
        num_classes=10,
        root="shared_fs/ensembles/data/imagewoof2",
        dataset_class=MultiTransImageFolder,
        target_transform_paths=IMAGEWOOF2_IDX_MAP_DICT,
    ),
    "imagewoof2-160": DatasetMetadata(
        num_classes=10,
        root="shared_fs/ensembles/data/imagewoof2-160",
        dataset_class=MultiTransImageFolder,
        target_transform_paths=IMAGEWOOF2_IDX_MAP_DICT,
    ),
    "imagewoof2-320": DatasetMetadata(
        num_classes=10,
        root="shared_fs/ensembles/data/imagewoof2-320",
        dataset_class=MultiTransImageFolder,
        target_transform_paths=IMAGEWOOF2_IDX_MAP_DICT,
    ),
    "imagenette2": DatasetMetadata(
        num_classes=10,
        root="shared_fs/ensembles/data/imagenette2",
        dataset_class=MultiTransImageFolder,
        target_transform_paths=IMAGENETTE2_IDX_MAP_DICT,
    ),
    "imagenette2-160": DatasetMetadata(
        num_classes=10,
        root="shared_fs/ensembles/data/imagenette2-160",
        dataset_class=MultiTransImageFolder,
        target_transform_paths=IMAGENETTE2_IDX_MAP_DICT,
    ),
    "imagenette2-320": DatasetMetadata(
        num_classes=10,
        root="shared_fs/ensembles/data/imagenette2-320",
        dataset_class=MultiTransImageFolder,
        target_transform_paths=IMAGENETTE2_IDX_MAP_DICT,
    ),
}


def build_target_transforms(path: Optional[str] = None) -> Union[Callable, None]:
    if path is None:
        return
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    label_transform = lambda idx: mapping[idx]
    return label_transform


def get_dataset(
    dataset_name: str,
    split: Literal["train", "val", "test"],
    transforms: Union[Callable, List[Callable]] = None,
) -> Dataset:
    dataset_metadata = DATASET_METADATA_BY_NAME[dataset_name].to_attrdict()
    if dataset_metadata.dataset_class == MultiTransImageFolder:
        root = dataset_metadata.root + "/" + split
        target_transform = build_target_transforms(dataset_metadata.target_transform_paths[split])
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


def build_timm_transforms(model_name: str) -> Callable:
    """Returns a list of timm transforms from a list of timm model names."""
    transform_keys = [arg for arg in signature(timm.data.create_transform).parameters.keys()]
    cfg = timm.models.get_pretrained_cfg(model_name)
    transform_kwargs = {key: cfg[key] for key in transform_keys if key in cfg}
    transform = timm.data.create_transform(is_training=False, **transform_kwargs)
    return transform
