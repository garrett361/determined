from typing import List, Optional

import pandas as pd
import timm
import torch.nn as nn

ALL_MODELS_DF = pd.read_feather("models/selected_models.feather").set_index("model")


def get_model_names() -> List[str]:
    return ALL_MODELS_DF.index.to_list()


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
