import torch
import torch.nn as nn

from .tDLGM import tDLGM, device, tDLGMCrossEntropy



# ── Time Recognition ──────────────────────────────────────────────────────────

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
        # Keep LSTM-compatible (h, c) state shape expected by Generator.
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



# ── tDLGM ─────────────────────────────────────────────────────────────────────

class tDLGMAttention(tDLGM):
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


class tDLGMAttentionCrossEntropy(tDLGMCrossEntropy):
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




# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.optim import Adam

    model = tDLGMAttention(
        input_dim=10,
        hidden_size=20,
        latent_dim=5,
        output_dim=10,
        layers=2,
        seq_len=3,
        num_heads=2,
        device=device,
    ).to(device)
    optimizer = Adam(model.get_parameters(), lr=0.1)

    x = torch.randn(50, 3, 10).to(device)  ## used for the state recognition
    y = torch.randn(50, 1, 10).to(device)  ## the value to be reconstructed
    x_1 = torch.cat((x, y), dim=1)[:, 1:, :]  ## used for the recognition

    before = model.get_loss(x, x_1, y)
    for _ in range(300):
        loss = model.train_step(x, x_1, y, optimizer)
    after = model.get_loss(x, x_1, y)
    print(f"Loss before training: {before}")
    print(f"Loss after training: {after}")
    assert after < before, "Loss did not decrease after training! MSELoss"

    model = tDLGMAttentionCrossEntropy(
        input_dim=10,
        hidden_size=20,
        latent_dim=5,
        output_dim=10,
        layers=2,
        seq_len=3,
        device=device,
    ).to(device)
    optimizer = Adam(model.get_parameters(), lr=0.1)

    x = torch.randn(50, 3, 10).to(device)  ## used for the state recognition
    y = torch.randint(0, 10, (50, 1)).to(device)  ## the value to be reconstructed (class labels)
    x_1 = torch.cat((x, nn.functional.one_hot(y.squeeze(), num_classes=10).float().unsqueeze(1)), dim=1)[:, 1:, :]  ## used for the recognition

    before = model.get_loss(x, x_1, y)
    for _ in range(300):
        loss = model.train_step(x, x_1, y, optimizer)
    after = model.get_loss(x, x_1, y)
    print(f"Loss before training: {before}")
    print(f"Loss after training: {after}")
    assert after < before, "Loss did not decrease after training! CrossEntropyLoss"



