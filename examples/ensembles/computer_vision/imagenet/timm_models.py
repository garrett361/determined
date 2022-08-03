"""From timm==0.6.5, we took the top-ten performing model with pre-trained weights as measured
by ImageNet validation accuracy (top-1), as well as the top-ten smallest models which achieved at
least 75% accuracy on the same task.
"""

import itertools
import logging
import math
import random
from typing import Literal, List

import pandas as pd
import timm
import torch.nn as nn

logging.basicConfig(level=logging.DEBUG, format=det.LOG_FORMAT)

SMALL_TIMM_MODELS_DF = pd.read_feather("small_timm_models.feather").set_index("model")
TOP_TIMM_MODELS_DF = pd.read_feather("top_timm_models.feather").set_index("model")
ALL_MODELS_DF = pd.concat([SMALL_TIMM_MODELS_DF, TOP_TIMM_MODELS_DF])

model_criteria_map = {
    "small": SMALL_TIMM_MODELS_DF,
    "top": TOP_TIMM_MODELS_DF,
    "all": ALL_MODELS_DF,
}


def get_model_names_from_criteria(model_criteria: Literal["top", "small", "all"]) -> List[str]:
    return model_criteria_map[model_criteria].index.to_list()


def get_timm_ensembles_of_model_names(
    model_criteria: Literal["top", "small", "all"],
    num_base_models: int,
    num_ensembles: int,
    seed: int = 42,
    offset: int = 0,
) -> List[List[str]]:
    """Returns num_ensembles unique ensembles of timm model names, each comprising of
    num_base_models models.  Use num_base_models = -1 to use all possible base models and/or
    num_ensembles = -1 to get all possible ensembles.
    """
    try:
        base_models = model_criteria_map[model_criteria].index.to_list()
    except KeyError:
        raise ValueError(f"Unknown model_criteria: {model_criteria}")
    assert (
        len(base_models) >= num_base_models
    ), f"num_base_models cannot be greater than {len(base_models)}, the number of base models."
    if num_base_models == -1:
        num_base_models = len(base_models)
    if num_ensembles == -1:
        ensembles = list(itertools.combinations(base_models, num_base_models))
    else:
        max_ensembles = math.comb(len(base_models), num_base_models)
        assert num_ensembles + offset <= max_ensembles, (
            f"num_ensembles (plus the offset of {offset}) is greater than {max_ensembles}, the"
            f"maximum number possible ensembles of size {num_base_models} drawn from"
            f" {len(base_models)} options."
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


def build_timm_models(model_names: List[str], checkpoint_path_prefix: str) -> List[nn.Module]:
    """Returns a list of models, each of which is a timm model."""
    models = []
    for name in model_names:
        model_data = ALL_MODELS_DF.loc[name]
        logging.info(f"Building model {name}...")
        checkpoint_path = checkpoint_path_prefix + model_data.state_dict_path
        model = timm.create_model(name, checkpoint_path=checkpoint_path)
        models.append(model)
    return models
