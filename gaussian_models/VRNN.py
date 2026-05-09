from collections.abc import Iterator

import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class VRNN(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        device=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layers = layers
        self.seq_len = seq_len
        self.device = device

        self.phi_x = nn.Sequential(
            nn.Linear(input_dim, hidden_size, device=device),
            nn.ReLU(),
        )
        self.phi_z = nn.Sequential(
            nn.Linear(latent_dim, hidden_size, device=device),
            nn.ReLU(),
        )

        self.enc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size, device=device),
            nn.ReLU(),
        )
        self.enc_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_size, latent_dim, device=device),
            nn.Softplus(),
        )

        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, device=device),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_size, latent_dim, device=device),
            nn.Softplus(),
        )

        self.dec = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size, device=device),
            nn.ReLU(),
        )
        self.dec_mean = nn.Linear(hidden_size, output_dim, device=device)

        self.rnn = nn.GRU(
            input_size=hidden_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
        )

        self.mse = nn.MSELoss()

    def get_parameters(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def _reparameterized_sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(std)
        return mean + eps * std

    def _kld_gauss(
        self,
        mean_post: torch.Tensor,
        std_post: torch.Tensor,
        mean_prior: torch.Tensor,
        std_prior: torch.Tensor,
    ) -> torch.Tensor:
        post_var = std_post.pow(2) + EPS
        prior_var = std_prior.pow(2) + EPS
        kld = (
            torch.log(prior_var)
            - torch.log(post_var)
            + (post_var + (mean_post - mean_prior).pow(2)) / prior_var
            - 1.0
        )
        return 0.5 * torch.sum(kld)

    def _forward_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(self.layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        kld_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        decoded = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            phi_x_t = self.phi_x(x_t)

            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], dim=1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) + EPS

            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t) + EPS

            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], dim=1))
            dec_mean_t = self.dec_mean(dec_t)
            decoded.append(dec_mean_t)

            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(1), h)

            kld_loss = kld_loss + self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

        return kld_loss, torch.stack(decoded, dim=1)

    def _loss(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        kld_loss: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        target = y.reshape_as(y_hat)
        recon = self.mse(y_hat, target)
        kld = kld_loss / (batch_size * max(seq_len, 1))
        return recon + kld

    def get_loss(self, x, x_1, y) -> float:
        with torch.no_grad():
            return self.train_step(x, x_1, y, optimizer=None)

    def train_step(self, x, x_1, y, optimizer) -> float:
        if optimizer is not None:
            optimizer.zero_grad()

        kld_loss, decoded = self._forward_sequence(x_1)
        pred = decoded[:, -1, :].unsqueeze(1)
        loss = self._loss(y, pred, kld_loss, x_1.size(0), x_1.size(1))

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        return loss.item()

    def forward(self, x) -> torch.Tensor:
        _, decoded = self._forward_sequence(x)
        return decoded[:, -1, :].unsqueeze(1)


# Backward-compatible alias for the old typo'd class name.
VRNNN = VRNN
