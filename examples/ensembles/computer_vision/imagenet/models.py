"""From timm==0.6.5, we took the top-ten performing model with pre-trained weights as measured
by ImageNet validation accuracy (top-1), as well as the top-ten smallest models which achieved at
least 75% accuracy on the same task.
"""

from typing import Literal, List
import random


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

base_models = {"best": timm_best_models, "smallest": timm_smallest_models_75_top1_masked}


def get_timm_model_ensembles(
    criteria: Literal["best", "smallest"], num_base_models: int, num_ensembles: int, seed: int = 42
) -> List[List[str]]:
    """Returns num_ensembles unique ensembles of timm model names, each comprising of
    num_base_models models.
    """
    if criteria == "best":
        base_models = timm_best_models
    elif criteria == "smallest":
        base_models = timm_smallest_models_75_top1_masked
    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    random.seed(seed)
    ensembles = []
    for _ in range(num_ensembles):
        while True:
            new_ensemble = sorted(random.sample(base_models, k=num_base_models))
            if new_ensemble not in ensembles:
                ensembles.append(new_ensemble)
                break
    return ensembles
