"""From timm==0.6.5, we took the top-ten performing model with pre-trained weights as measured
by ImageNet validation accuracy (top-1), as well as the top-ten smallest models which achieved at
least 75% accuracy on the same task.
"""

import itertools
import math
from typing import Literal, List, Union
import random

import timm
import torch.nn as nn


timm_best_models = [
    "beit_large_patch16_512",
    "beit_large_patch16_384",
    "tf_efficientnet_l2_ns",
    "tf_efficientnet_l2_ns_475",
    "convnext_xlarge_384_in22ft1k",
    "beit_large_patch16_224",
    "convnext_large_384_in22ft1k",
    "swin_large_patch4_window12_384",
    "vit_large_patch16_384",
    "volo_d5_512",
]

timm_smallest_models_75_top1_masked = [
    "xcit_nano_12_p16_384_dist",
    "xcit_nano_12_p8_384_dist",
    "xcit_nano_12_p8_224_dist",
    "semnasnet_100",
    "mixnet_s",
    "tf_mixnet_s",
    "mobilenetv2_110d",
    "efficientnet_lite0",
    "rexnet_100",
    "mixnet_m",
]


def get_timm_ensembles_of_model_names(
    model_criteria: Literal["best", "smallest"],
    num_base_models: int,
    num_ensembles: int,
    seed: int = 42,
    offset: int = 0,
) -> List[List[str]]:
    """Returns num_ensembles unique ensembles of timm model names, each comprising of
    num_base_models models.  Use num_ensembles = -1 to get all possible ensembles.
    """
    if model_criteria == "best":
        base_models = timm_best_models
    elif model_criteria == "smallest":
        base_models = timm_smallest_models_75_top1_masked
    else:
        raise ValueError(f"Unknown model_criteria: {model_criteria}")
    assert (
        len(base_models) >= num_base_models
    ), f"num_base_models cannot be greater than {len(base_models)}, the number of base models."
    if num_ensembles == -1:
        ensembles = list(itertools.combinations(base_models, num_base_models))
    else:
        max_ensembles = math.comb(len(base_models), num_base_models)
        assert num_ensembles <= max_ensembles, (
            f"num_ensembles is greater than {max_ensembles}, the maximum number possible ensembles of"
            f" size {num_base_models} drawn from {len(base_models)} options."
        )

        random.seed(seed)
        ensembles = []
        for _ in range(num_ensembles + offset):
            while True:
                new_ensemble = sorted(random.sample(base_models, k=num_base_models))
                if new_ensemble not in ensembles:
                    ensembles.append(new_ensemble)
                    break
    return ensembles[offset:]


def build_timm_model_list(model_names: List[str], pretrained: bool = True) -> List[nn.Module]:
    """Returns a list of models, each of which is a timm model."""
    models = []
    for model_name in model_names:
        print(f"Building model {model_name}...")
        model = timm.create_model(model_name, pretrained=pretrained)
        models.append(model)
    return models
