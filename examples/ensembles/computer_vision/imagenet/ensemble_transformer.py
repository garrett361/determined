import random
from typing import List, Literal, Optional

import torch
import torch.nn as nn

import timm_models


class EnsembleTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        head_dim, remainder = divmod(d_model, num_heads)
        assert not remainder, "d_model must be divisible by num_heads"
        self.ln1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.ma = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True, **factory_kwargs
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model, **factory_kwargs),
        )
        self.ln2 = nn.LayerNorm(d_model, **factory_kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        k = q = v = self.ln1(inputs)
        x = self.ma(k, q, v, need_weights=False)[0] + inputs
        x = self.ffn(self.ln2(x)) + x
        return x


class EnsembleTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        d_model: int = 1000,
        num_heads: int = 2,
        dim_feedforward: int = 2048,
        init_transformer_scale: float = 0.1,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        # Initialize as scale parameter which initializes the balance between the naive and
        # transformer outputs.
        self.init_transformer_scale = nn.Parameter(
            torch.tensor(init_transformer_scale, **factory_kwargs)
        )
        self.layers = nn.ModuleList(
            [
                EnsembleTransformerLayer(d_model, num_heads, dim_feedforward, **factory_kwargs)
                for _ in range(num_layers)
            ]
        )

    def forward(self, model_logits: torch.Tensor) -> torch.Tensor:
        # model_logits is expected to be a (B, C, M)-sized tensor
        # with the three dimensions specifying the batch, class, and model
        # respectively.  Transformers expect position to be at the 1-dimension
        # and so we re-shape to (B, M, C) before passing through the transformer.

        naive_model_logits = model_logits.mean(dim=-1)
        reshaped_model_logits = model_logits.transpose(-2, -1)
        # Append class token to the start of the input
        batch_size = model_logits.shape[0]
        expanded_class_token = self.class_token.repeat(batch_size, 1, 1)
        x = torch.cat([expanded_class_token, reshaped_model_logits], dim=1)

        for layer in self.layers:
            x = layer(x)
        output_class_token = x[:, 0]
        return naive_model_logits + self.init_transformer_scale * output_class_token


class Mixer(nn.Module):
    def __init__(
        self,
        dim: int,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.dim = dim
        self.mixed_idxs = None
        self.unmixed_idxs = None
        self.device = device

    def forward(self, inputs: torch.Tensor, action: Literal["mix", "unmix"]) -> torch.Tensor:
        if not self.training:
            return inputs
        if action == "mix":
            dim_size = inputs.shape[self.dim]
            self.mixed_idxs = torch.randperm(dim_size, device=self.factory_kwargs["device"])
            self.unmixed_idxs = self.mixed_idxs.topk(len(self.mixed_idxs), largest=False).indices
            outputs = inputs.index_select(dim=self.dim, index=self.mixed_idxs)
        elif action == "unmix":
            outputs = inputs.index_select(dim=self.dim, index=self.unmixed_idxs)
        return outputs


class ArgsKwargsIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return inputs


class TimmModelEnsembleTransformer(nn.Module):
    def __init__(
        self,
        model_names: List[str],
        checkpoint_path_prefix: str,
        num_layers: int = 2,
        d_model: int = 1000,
        num_heads: int = 2,
        dim_feedforward: int = 2048,
        mix_models=False,
        mix_classes=False,
        model_dim: int = -1,
        class_dim: int = 1,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Don't use nn.ModuleList because we don't need to track the model params in the state_dict,
        # as they're all pre-trained. This also keeps all models in self.models in the .eval()
        # state, even if self.train() is called.
        self.models = timm_models.build_timm_models(
            model_names=model_names, checkpoint_path_prefix=checkpoint_path_prefix, device=device
        )
        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        self.model_mixer = (
            ArgsKwargsIdentity() if not mix_models else Mixer(dim=model_dim, **factory_kwargs)
        )
        self.class_mixer = (
            ArgsKwargsIdentity() if not mix_classes else Mixer(dim=class_dim, **factory_kwargs)
        )

        self.ensemble_transformer = EnsembleTransformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            **factory_kwargs
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Expects inputs to be a list of tensors, one for each model in self.models with the
        # expected model-specific transform applied
        model_logits = torch.stack(
            [model(inpt) for model, inpt in zip(self.models, inputs)], dim=-1
        )
        model_logits = self.model_mixer(model_logits, action="mix")
        model_logits = self.class_mixer(model_logits, action="mix")
        ensemble_logits = self.ensemble_transformer(model_logits)
        ensemble_logits = self.class_mixer(ensemble_logits, action="unmix")
        return ensemble_logits
