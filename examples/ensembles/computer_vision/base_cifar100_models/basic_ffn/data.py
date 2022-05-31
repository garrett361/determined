import dataclasses
from io import BytesIO, StringIO
import json
from typing import Any, Dict, Tuple, Union
import os
import urllib

import attrdict
from determined.util import download_gcs_blob_with_backoff
import filelock
from google.cloud import storage
from PIL import Image as PILImage
from timm.data import create_transform
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision

ImageStat = Union[Tuple[float], Tuple[float, float, float]]


@dataclasses.dataclass
class DatasetMetadata:
    num_classes: int
    img_size: int
    in_chans: int
    mean: ImageStat
    std: ImageStat
    labels: Tuple[str]

    def to_dict(self) -> Dict[str, Union[int, ImageStat]]:
        return dataclasses.asdict(self)


# Build the list of ImageNet classes.
with urllib.request.urlopen(
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
) as url:
    imagenet_json_data = json.loads(url.read().decode())
    imagenet_classes_list = sorted(
        [(int(idx), name) for idx, (_, name) in imagenet_json_data.items()]
    )
    imagenet_classes_tuple = tuple(name for _, name in imagenet_classes_list)

DATASET_METADATA_BY_NAME = {
    "mnist": DatasetMetadata(
        num_classes=10,
        img_size=28,
        in_chans=1,
        mean=(0.1307,),
        std=(0.3081,),
        labels=tuple(str(n) for n in range(10)),
    ),
    "cifar10": DatasetMetadata(
        num_classes=10,
        img_size=32,
        in_chans=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        labels=(
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ),
    ),
    "cifar100": DatasetMetadata(
        num_classes=100,
        img_size=32,
        in_chans=3,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        labels=(
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "cra",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ),
    ),
    "imagenet": DatasetMetadata(
        num_classes=1000,
        img_size=224,
        in_chans=3,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        labels=imagenet_classes_tuple,
    ),
}


class GCSImageNetStreamDataset(Dataset):
    """Streams ImageNet images from Google Cloud Storage into memory. Adapted from byol example."""

    def __init__(
        self,
        data_config: attrdict.AttrDict,
        train: bool,
        transform: nn.Module,
    ) -> None:
        """
        Args:
            data_config (attrdict.AttrDict): AttrDict containing 'gcs_bucket', 'gcs_train_blob_list_path', and
                'gcs_validation_blob_list_path' keys.
            train (bool): flag for building the training (True) or validation (False) datasets.
            transform (nn.Module): transforms to be applied to the dataset.
        """
        self._transform = transform
        self._storage_client = storage.Client()
        self._bucket = self._storage_client.bucket(data_config.gcs_bucket)
        # When the dataset is first initialized, we'll loop through to catalogue the classes (subdirectories)
        # This step might take a long time.
        self._imgs_paths = []
        self._labels = []
        self._subdir_to_class: Dict[str, int] = {}
        class_count = 0
        if train:
            blob_list_path = data_config.gcs_train_blob_list_path
        else:
            blob_list_path = data_config.gcs_validation_blob_list_path
        blob_list_blob = self._bucket.blob(blob_list_path)
        blob_list_io = StringIO(
            download_gcs_blob_with_backoff(
                blob_list_blob, n_retries=4, max_backoff=2
            ).decode("utf-8")
        )
        blob_list = [s.strip() for s in blob_list_io.readlines()]
        for path in blob_list:
            self._imgs_paths.append(path)
            sub_dir = path.split("/")[-2]
            if sub_dir not in self._subdir_to_class:
                self._subdir_to_class[sub_dir] = class_count
                class_count += 1
            self._labels.append(self._subdir_to_class[sub_dir])
        dataset_str = "training" if train else "validation"
        print(f"The {dataset_str} dataset contains {len(self._imgs_paths)} records.")

    def __len__(self) -> int:
        return len(self._imgs_paths)

    def __getitem__(self, idx: int) -> Tuple[PILImage.Image, int]:
        img_path = self._imgs_paths[idx]
        blob = self._bucket.blob(img_path)
        img_str = download_gcs_blob_with_backoff(blob)
        img_bytes = BytesIO(img_str)
        img = PILImage.open(img_bytes)
        img = img.convert("RGB")
        return self._transform(img), self._labels[idx]


DATASET_DICT = {
    "mnist": torchvision.datasets.MNIST,
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
    "imagenet": GCSImageNetStreamDataset,
}


def get_dataset(
    data_config: attrdict.AttrDict, train: bool, transform: nn.Module
) -> Dataset:
    """
    Downloads or streams (in the case of ImageNet) the training or validation dataset, and applies `transform`
    to the corresponding images.
    """
    dataset_name = data_config.dataset_name
    dataset = DATASET_DICT[dataset_name]
    if dataset_name == "imagenet":
        # Imagenet data is streamed from GCS directly into memory.
        return dataset(data_config=data_config, train=train, transform=transform)
    else:
        download_dir = data_config.download_dir
        os.makedirs(download_dir, exist_ok=True)
        with filelock.FileLock(os.path.join(download_dir, "lock")):
            return dataset(
                root=download_dir,
                train=train,
                download=True,
                transform=transform,
            )


def build_transform(
    dataset_metadata: Any, transform_config: attrdict.AttrDict, train: bool
) -> nn.Module:
    """Generate transforms via timm's transform factory."""
    return create_transform(
        input_size=dataset_metadata.img_size,
        is_training=train,
        mean=dataset_metadata.mean,
        std=dataset_metadata.std,
        **transform_config,
    )
