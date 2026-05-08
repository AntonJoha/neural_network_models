import torch
import torch.nn as nn

from .tDLGM import tDLGM


class AttentionTimeLayer(nn.Module):
    def __init__(self, input_dim=1, hidden_size=1, num_heads=1, device=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size, device=device)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            batch_first=True,
            device=device,
        )
        self.norm = nn.LayerNorm(hidden_size, device=device)

    def forward(self, x):
        h = self.input_proj(x)
        attn_out, _ = self.attention(h, h, h, need_weights=False)
        h = self.norm(h + attn_out)
        hidden_state = h[:, -1, :].unsqueeze(0)
        cell_state = torch.zeros_like(hidden_state)
        return hidden_state, cell_state


class AttentionTimeRecognition(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        layers=1,
        num_heads=1,
        device=None,
    ):
        super().__init__()
        self.time_layers = nn.ModuleList(
            AttentionTimeLayer(
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_heads=num_heads,
                device=device,
            )
            for _ in range(layers)
        )

    def forward(self, x):
        return [layer(x) for layer in self.time_layers]


class tDLGM_attention(tDLGM):
    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        num_heads=1,
        device=None,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            output_dim=output_dim,
            layers=layers,
            seq_len=seq_len,
            device=device,
        )
        self.model_t = AttentionTimeRecognition(
            input_dim=input_dim,
            hidden_size=hidden_size,
            layers=layers,
            num_heads=num_heads,
            device=device,
        )
