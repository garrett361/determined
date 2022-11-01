import itertools
import math
import random
from typing import Literal, List, Optional

import pandas as pd
import timm
import torch.nn as nn

SMALL_TIMM_MODELS_DF = pd.read_feather("models/small_timm_models.feather").set_index("model")
TOP_TIMM_MODELS_DF = pd.read_feather("models/top_timm_models.feather").set_index("model")
ALL_MODELS_DF = pd.concat([SMALL_TIMM_MODELS_DF, TOP_TIMM_MODELS_DF])

model_criteria_map = {
    "small": SMALL_TIMM_MODELS_DF,
    "top": TOP_TIMM_MODELS_DF,
    "all": ALL_MODELS_DF,
}


def get_model_names_from_criteria(model_criteria: Literal["top", "small", "all"]) -> List[str]:
    return model_criteria_map[model_criteria].index.to_list()


def build_timm_model(
    model_name: str, checkpoint_path_prefix: str, device: Optional[str] = None
) -> nn.Module:
    """Returns a list of models, each of which is a timm model."""
    model_data = ALL_MODELS_DF.loc[model_name]
    print(f"Building model {model_name}...")
    checkpoint_path = checkpoint_path_prefix + model_data.state_dict_path
    model = timm.create_model(model_name, checkpoint_path=checkpoint_path)
    model.to(device)
    return model
