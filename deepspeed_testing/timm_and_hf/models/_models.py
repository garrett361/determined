from typing import List, Optional

import pandas as pd
import timm
import torch.nn as nn

TIMM_MODELS_DF = pd.read_feather("models/selected_timm_models.feather").set_index("model")
HF_GLUE_MODELS_DF = pd.read_feather("models/selected_hf_glue_models.feather").set_index("id")
HF_CLM_MODELS_DF = pd.read_feather("models/selected_hf_clm_models.feather").set_index("id")

TASK_TO_DF_MAP = {"timm": TIMM_MODELS_DF, "hf_glue": HF_GLUE_MODELS_DF, "hf_clm": HF_CLM_MODELS_DF}


def get_model_names(task: str) -> List[str]:
    return TASK_TO_DF_MAP[task].index.to_list()


def build_timm_model(
    model_name: str, checkpoint_path_prefix: str, device: Optional[str] = None
) -> nn.Module:
    """Returns a list of models, each of which is a timm model."""
    model_data = TIMM_MODELS_DF.loc[model_name]
    print(f"Building model {model_name}...")
    checkpoint_path = checkpoint_path_prefix + model_data.state_dict_path
    model = timm.create_model(model_name, checkpoint_path=checkpoint_path)
    model.to(device)
    return model
