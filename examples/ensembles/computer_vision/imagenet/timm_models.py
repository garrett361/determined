"""From timm==0.6.5, we took the top-ten performing model with pre-trained weights as measured
by ImageNet validation accuracy (top-1), as well as the top-ten smallest models which achieved at
least 75% accuracy on the same task.
"""

import itertools
import math
import random
from typing import Generator, Literal, List

import pandas as pd
import timm
import torch.nn as nn


SMALL_TIMM_MODELS_DF = pd.read_feather("small_timm_models.feather").set_index("model")
TOP_TIMM_MODELS_DF = pd.read_feather("top_timm_models.feather").set_index("model")

ALL_MODELS_DF = pd.concat([SMALL_TIMM_MODELS_DF, TOP_TIMM_MODELS_DF])

MODEL_CRITERIA_MAP = {
    "small": SMALL_TIMM_MODELS_DF,
    "top": TOP_TIMM_MODELS_DF,
    "all": ALL_MODELS_DF,
}


class TimmEnsembleGenerator:
    def __init__(
        self,
        model_criteria: Literal["top", "small", "all"],
        num_base_models: int,
        num_ensembles: int,
        seed: int = 42,
    ) -> None:
        self.models_df = MODEL_CRITERIA_MAP[model_criteria]
        assert len(self.models_df) >= num_base_models, (
            f"num_base_models cannot be greater than {len(self.models_df)},"
            f" the number of base models."
        )
        self.num_base_models = num_base_models
        self.num_ensembles = num_ensembles
        self.seed = seed

    def get_model_names(self) -> List[str]:
        return list(self.models_df.index)

    def get_ensembles_of_model_names(self, offset: int = 0) -> List[List[str]]:
        """Returns num_ensembles unique ensembles of timm model names, each comprising of
        num_base_models models.  Use num_base_models = -1 to use all possible base models and/or
        num_ensembles = -1 to get all possible ensembles.
        """
        if self.num_base_models == -1:
            self.num_base_models = len(self.models_df)
        if self.num_ensembles == -1:
            ensembles = list(itertools.combinations(self.models_df, self.num_base_models))
        else:
            max_ensembles = math.comb(len(self.models_df), self.num_base_models)
            assert self.num_ensembles + offset <= max_ensembles, (
                f"self.num_ensembles (plus the offset of {offset}) is greater than {max_ensembles},"
                f" the maximum number possible ensembles of size {self.num_base_models} drawn from"
                f" {len(self.models_df)} options."
            )

            random.seed(self.seed)
            ensembles = []
            for _ in range(self.num_ensembles + offset):
                while True:
                    new_ensemble = sorted(
                        random.sample(self.models_df.index.to_list(), k=self.num_base_models)
                    )
                    if new_ensemble not in ensembles:
                        ensembles.append(new_ensemble)
                        break
        return ensembles[offset:]

    @staticmethod
    def model_list_from_model_names(
        self, model_names: List[str], checkpoint_path_prefix: str = ""
    ) -> List[nn.Module]:
        """Returns a generator of timm models, one for each model name."""
        model_list = []
        for model_name in model_names:
            state_dict_path = self.models_df.loc[model_name].state_dict_path
            checkpoint_path = checkpoint_path_prefix + state_dict_path
            model = timm.create_model(model_name, checkpoint_path=checkpoint_path)
            model_list.append(model)
        return model
