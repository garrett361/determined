"""From timm==0.6.5, we took the top-ten performing model with pre-trained weights as measured
by ImageNet validation accuracy (top-1), as well as the top-ten smallest models which achieved at
least 75% accuracy on the same task.
"""

import itertools
import math
import random
from typing import Literal, List, Union
import pickle

import timm
import torch.nn as nn


with open("small_timm_models.pkl", "rb") as f:
    small_timm_models = pickle.load(f)
with open("top_timm_models.pkl", "rb") as f:
    top_timm_models = pickle.load(f)

all_models = {**small_timm_models, **top_timm_models}


def get_timm_ensembles_of_model_names(
    model_criteria: Literal["top", "small"],
    num_base_models: int,
    num_ensembles: int,
    seed: int = 42,
    offset: int = 0,
) -> List[List[str]]:
    """Returns num_ensembles unique ensembles of timm model names, each comprising of
    num_base_models models.  Use num_base_models = -1 to use all possible base models and/or
    num_ensembles = -1 to get all possible ensembles.
    """
    if model_criteria == "top":
        base_models = top_timm_models
    elif model_criteria == "small":
        base_models = small_timm_models
    else:
        raise ValueError(f"Unknown model_criteria: {model_criteria}")
    assert (
        len(base_models) >= num_base_models
    ), f"num_base_models cannot be greater than {len(base_models)}, the number of base models."
    if num_base_models == -1:
        num_base_models = len(base_models)
    if num_ensembles == -1:
        ensembles = list(itertools.combinations(base_models.keys(), num_base_models))
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
                new_ensemble = sorted(random.sample(base_models.keys(), k=num_base_models))
                if new_ensemble not in ensembles:
                    ensembles.append(new_ensemble)
                    break
    return ensembles[offset:]


def build_timm_model_list(model_names: List[str], checkpoint_path_prefix: str) -> List[nn.Module]:
    """Returns a list of models, each of which is a timm model."""
    models = []
    for model_name in model_names:
        print(f"Building model {model_name}...")
        checkpoint_path = checkpoint_path_prefix + all_models[model_name]
        print(checkpoint_path)
        model = timm.create_model(model_name, checkpoint_path=checkpoint_path)
        models.append(model)
    return models
