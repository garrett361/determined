import torch
import torch.nn as nn


class EnsembleTransformerLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, dim_feedforward: int, **factory_kwargs
    ) -> None:
        super().__init__()
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
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.layers = nn.ModuleList(
            [
                EnsembleTransformerLayer(d_model, num_heads, dim_feedforward, **factory_kwargs)
                for _ in range(num_layers)
            ]
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        mean_logits = logits.mean(dim=1)
        # Append class token to the start of the input
        batch_size = logits.shape[0]
        expanded_class_token = self.class_token.repeat(batch_size, 1, 1)
        x = torch.cat([expanded_class_token, logits], dim=1)
        for layer in self.layers:
            x = layer(x)
        output_class_token = x[:, 0]
        return mean_logits + output_class_token
