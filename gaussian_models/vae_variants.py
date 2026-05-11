from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn

from .DLGM import DLGM
from .tDLGM import tDLGM
from .VRNN import EPS, VRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _diag_kl(
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


def _gaussian_log_prob(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    var = std.pow(2) + EPS
    return -0.5 * (
        torch.log(2 * torch.pi * var)
        + (z - mean).pow(2) / var
    ).sum(dim=-1)


class BetaDLGM(DLGM):
    def __init__(self, *args, beta: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def _loss(self, y, y_hat, mean, R) -> torch.Tensor:
        epsilon = max(1e-6, torch.finfo(y_hat.dtype).eps)
        if y.numel() != y_hat.numel():
            raise ValueError(f"Target shape {tuple(y.shape)} is incompatible with prediction shape {tuple(y_hat.shape)}")
        target = y.reshape_as(y_hat)
        recon = self.mse(y_hat, target)
        matrix_size = mean[0].size(0) * mean[0].size(1)

        kl = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
        eye = torch.eye(R[0].size(-1), device=R[0].device, dtype=R[0].dtype).expand_as(R[0])
        for m, r in zip(mean, R, strict=True):
            c = r @ r.transpose(-2, -1)
            _, logdet = torch.linalg.slogdet(c + epsilon * eye)
            kl = kl + (
                0.5
                * torch.sum(
                    m.pow(2).sum(-1)
                    + c.diagonal(dim1=-2, dim2=-1).sum(-1)
                    - logdet
                    - 1
                )
                / matrix_size
            )
        return recon + self.beta * kl


class BetatDLGM(tDLGM):
    def __init__(self, *args, beta: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def _loss(self, y, y_hat, mean, R, s, t_1, reg) -> torch.Tensor:
        target = y.reshape_as(y_hat)
        recon = self.mse(y_hat, target)
        matrix_size = mean[0].size(0) * mean[0].size(1)

        kl = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
        for m, r in zip(mean, R, strict=True):
            c = r @ r.transpose(-2, -1)
            det = c.det().clamp(min=EPS)
            kl = kl + (
                0.5
                * torch.sum(
                    m.pow(2).sum(-1)
                    + c.diagonal(dim1=-2, dim2=-1).sum(-1)
                    - det.log()
                    - 1
                )
                / matrix_size
            )

        state_reg = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
        amount = len(s) * len(s[0])
        for a, b in zip(s, t_1, strict=True):
            state_reg = state_reg + reg * (self.mse(a[0], b[0]) + self.mse(a[1], b[1])) / amount

        return recon + self.beta * kl + state_reg


class ConditionalVRNN(VRNN):
    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        cond_dim=1,
        device=None,
        activation_function=nn.ReLU,
    ):
        self.raw_input_dim = input_dim
        self.cond_dim = cond_dim
        super().__init__(
            input_dim=input_dim + cond_dim,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            output_dim=output_dim,
            layers=layers,
            seq_len=seq_len,
            device=device,
            activation_function=activation_function,
        )

    def _conditioned(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if condition is None:
            condition = torch.zeros(x.size(0), self.cond_dim, device=x.device, dtype=x.dtype)

        if condition.dim() == 2:
            condition = condition.unsqueeze(1).expand(-1, x.size(1), -1)
        if condition.size(0) != x.size(0) or condition.size(1) != x.size(1) or condition.size(2) != self.cond_dim:
            raise ValueError(
                "condition must have shape [batch, cond_dim] or [batch, seq_len, cond_dim] "
                f"with cond_dim={self.cond_dim}"
            )
        return torch.cat([x, condition], dim=-1)

    def get_loss(self, _x, x_1, y, condition: torch.Tensor | None = None) -> float:
        with torch.no_grad():
            return self._compute_loss(self._conditioned(x_1, condition), y).item()

    def train_step(self, _x, x_1, y, optimizer, condition: torch.Tensor | None = None) -> float:
        if optimizer is not None:
            optimizer.zero_grad()

        loss = self._compute_loss(self._conditioned(x_1, condition), y)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        return loss.item()

    def forward(self, x, condition: torch.Tensor | None = None) -> torch.Tensor:
        _, decoded = self._forward_sequence(self._conditioned(x, condition))
        return decoded[:, -1, :].unsqueeze(1)


class IWAEVRNN(VRNN):
    def __init__(self, *args, importance_samples: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.importance_samples = max(1, int(importance_samples))

    def _compute_loss(self, x_1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        losses = []
        for _ in range(self.importance_samples):
            kld_loss, decoded = self._forward_sequence(x_1)
            pred = decoded[:, -1, :].unsqueeze(1)
            losses.append(self._loss(y, pred, kld_loss, x_1.size(0), x_1.size(1)))
        loss_stack = torch.stack(losses)
        return -torch.logsumexp(-loss_stack, dim=0) + torch.log(
            torch.tensor(float(self.importance_samples), device=x_1.device, dtype=x_1.dtype)
        )


class LadderVAE(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_size=16,
        latent_dim=8,
        output_dim=1,
        layers=1,
        seq_len=1,
        device=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.layers = layers
        self.seq_len = seq_len
        self.device = device

        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size, device=device),
            nn.ReLU(),
        )
        self.enc1_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.enc1_std = nn.Sequential(nn.Linear(hidden_size, latent_dim, device=device), nn.Softplus())

        self.enc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, device=device),
            nn.ReLU(),
        )
        self.enc2_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.enc2_std = nn.Sequential(nn.Linear(hidden_size, latent_dim, device=device), nn.Softplus())

        self.prior1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_size, device=device),
            nn.ReLU(),
        )
        self.prior1_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.prior1_std = nn.Sequential(nn.Linear(hidden_size, latent_dim, device=device), nn.Softplus())

        self.dec = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim, device=device),
        )
        self.mse = nn.MSELoss()

    def get_parameters(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    @staticmethod
    def _sample(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(std) * std

    def _encode(self, x: torch.Tensor):
        h1 = self.enc1(x)
        mean1 = self.enc1_mean(h1)
        std1 = self.enc1_std(h1) + EPS

        h2 = self.enc2(h1)
        mean2 = self.enc2_mean(h2)
        std2 = self.enc2_std(h2) + EPS

        z2 = self._sample(mean2, std2)
        p1 = self.prior1(z2)
        prior1_mean = self.prior1_mean(p1)
        prior1_std = self.prior1_std(p1) + EPS

        enc_var1 = std1.pow(2) + EPS
        prior_var1 = prior1_std.pow(2) + EPS
        post_var1 = 1.0 / (1.0 / enc_var1 + 1.0 / prior_var1)
        post_mean1 = post_var1 * (mean1 / enc_var1 + prior1_mean / prior_var1)
        post_std1 = torch.sqrt(post_var1 + EPS)
        z1 = self._sample(post_mean1, post_std1)

        return z1, z2, post_mean1, post_std1, prior1_mean, prior1_std, mean2, std2

    def _compute_loss(self, x_1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_last = x_1[:, -1, :]
        z1, z2, post_mean1, post_std1, prior1_mean, prior1_std, mean2, std2 = self._encode(x_last)
        pred = self.dec(torch.cat([z1, z2], dim=-1)).unsqueeze(1)
        target = y.reshape_as(pred)

        recon = self.mse(pred, target)
        prior2_mean = torch.zeros_like(mean2)
        prior2_std = torch.ones_like(std2)
        kl2 = _diag_kl(mean2, std2, prior2_mean, prior2_std)
        kl1 = _diag_kl(post_mean1, post_std1, prior1_mean, prior1_std)
        return recon + (kl1 + kl2) / x_1.size(0)

    def get_loss(self, _x, x_1, y) -> float:
        with torch.no_grad():
            return self._compute_loss(x_1, y).item()

    def train_step(self, _x, x_1, y, optimizer) -> float:
        if optimizer is not None:
            optimizer.zero_grad()
        loss = self._compute_loss(x_1, y)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        return loss.item()

    def forward(self, x) -> torch.Tensor:
        x_last = x[:, -1, :]
        z1, z2, *_ = self._encode(x_last)
        return self.dec(torch.cat([z1, z2], dim=-1)).unsqueeze(1)


class KalmanVAE(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_size=16,
        latent_dim=8,
        output_dim=1,
        layers=1,
        seq_len=1,
        device=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = layers
        self.seq_len = seq_len
        self.device = device

        self.obs_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_size, device=device),
            nn.ReLU(),
        )
        self.obs_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.obs_std = nn.Sequential(nn.Linear(hidden_size, latent_dim, device=device), nn.Softplus())

        self.transition = nn.Linear(latent_dim, latent_dim, bias=False, device=device)
        nn.init.eye_(self.transition.weight)
        self.kalman_gain_logits = nn.Parameter(torch.zeros(latent_dim, device=device))

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim, device=device),
        )
        self.mse = nn.MSELoss()

    def get_parameters(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def _filter(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        state = torch.zeros(batch_size, self.latent_dim, device=x.device, dtype=x.dtype)
        gain = torch.sigmoid(self.kalman_gain_logits).view(1, -1)

        states, obs_means, obs_stds, preds = [], [], [], []
        for t in range(seq_len):
            obs_h = self.obs_enc(x[:, t, :])
            mean_t = self.obs_mean(obs_h)
            std_t = self.obs_std(obs_h) + EPS
            z_obs = mean_t + torch.randn_like(std_t) * std_t

            pred_state = self.transition(state)
            state = gain * z_obs + (1.0 - gain) * pred_state

            states.append(state)
            obs_means.append(mean_t)
            obs_stds.append(std_t)
            preds.append(pred_state)

        return torch.stack(states, dim=1), torch.stack(obs_means, dim=1), torch.stack(obs_stds, dim=1), torch.stack(preds, dim=1)

    def _compute_loss(self, x_1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        states, obs_means, obs_stds, preds = self._filter(x_1)
        pred_y = self.decoder(states[:, -1, :]).unsqueeze(1)
        target = y.reshape_as(pred_y)

        recon = self.mse(pred_y, target)
        prior_mean = preds.detach()
        prior_std = torch.ones_like(obs_stds)
        dyn_kl = _diag_kl(obs_means, obs_stds, prior_mean, prior_std) / (x_1.size(0) * x_1.size(1))
        trans_reg = self.mse(states[:, 1:, :], preds[:, 1:, :]) if x_1.size(1) > 1 else torch.zeros((), device=x_1.device, dtype=x_1.dtype)
        return recon + dyn_kl + 0.1 * trans_reg

    def get_loss(self, _x, x_1, y) -> float:
        with torch.no_grad():
            return self._compute_loss(x_1, y).item()

    def train_step(self, _x, x_1, y, optimizer) -> float:
        if optimizer is not None:
            optimizer.zero_grad()
        loss = self._compute_loss(x_1, y)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        return loss.item()

    def forward(self, x) -> torch.Tensor:
        states, *_ = self._filter(x)
        return self.decoder(states[:, -1, :]).unsqueeze(1)


class SRNN(nn.Module):
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
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.layers = layers
        self.seq_len = seq_len

        self.phi_x = nn.Sequential(
            nn.Linear(input_dim, hidden_size, device=device),
            nn.ReLU(),
        )
        self.phi_z = nn.Sequential(
            nn.Linear(latent_dim, hidden_size, device=device),
            nn.ReLU(),
        )

        self.enc = nn.Sequential(
            nn.Linear(2 * hidden_size + latent_dim, hidden_size, device=device),
            nn.ReLU(),
        )
        self.enc_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.enc_std = nn.Sequential(nn.Linear(hidden_size, latent_dim, device=device), nn.Softplus())

        self.prior = nn.Sequential(
            nn.Linear(hidden_size + latent_dim, hidden_size, device=device),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.prior_std = nn.Sequential(nn.Linear(hidden_size, latent_dim, device=device), nn.Softplus())

        self.dec = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim, device=device),
        )

        self.det_cell = nn.GRUCell(2 * hidden_size, hidden_size, device=device)
        self.mse = nn.MSELoss()

    def get_parameters(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def _forward_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        z_prev = torch.zeros(batch_size, self.latent_dim, device=x.device, dtype=x.dtype)

        decoded = []
        kld_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            x_t = x[:, t, :]
            phi_x_t = self.phi_x(x_t)
            phi_z_prev = self.phi_z(z_prev)

            enc_h = self.enc(torch.cat([phi_x_t, h, z_prev], dim=-1))
            enc_mean = self.enc_mean(enc_h)
            enc_std = self.enc_std(enc_h) + EPS

            prior_h = self.prior(torch.cat([h, z_prev], dim=-1))
            prior_mean = self.prior_mean(prior_h)
            prior_std = self.prior_std(prior_h) + EPS

            z_t = enc_mean + torch.randn_like(enc_std) * enc_std
            phi_z_t = self.phi_z(z_t)

            dec_t = self.dec(torch.cat([h, phi_z_t], dim=-1))
            decoded.append(dec_t)

            h = self.det_cell(torch.cat([phi_x_t, phi_z_prev], dim=-1), h)
            z_prev = z_t

            kld_loss = kld_loss + _diag_kl(enc_mean, enc_std, prior_mean, prior_std)

        return kld_loss, torch.stack(decoded, dim=1)

    def _compute_loss(self, x_1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        kld, decoded = self._forward_sequence(x_1)
        pred = decoded[:, -1, :].unsqueeze(1)
        recon = self.mse(pred, y.reshape_as(pred))
        return recon + kld / (x_1.size(0) * x_1.size(1))

    def get_loss(self, _x, x_1, y) -> float:
        with torch.no_grad():
            return self._compute_loss(x_1, y).item()

    def train_step(self, _x, x_1, y, optimizer) -> float:
        if optimizer is not None:
            optimizer.zero_grad()
        loss = self._compute_loss(x_1, y)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        return loss.item()

    def forward(self, x) -> torch.Tensor:
        _, decoded = self._forward_sequence(x)
        return decoded[:, -1, :].unsqueeze(1)


class PlanarFlow(nn.Module):
    def __init__(self, dim: int, device=None):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim, device=device) * 0.01)
        self.w = nn.Parameter(torch.randn(dim, device=device) * 0.01)
        self.b = nn.Parameter(torch.zeros((), device=device))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        z_new = z + activation.unsqueeze(-1) * self.u

        psi = (1.0 - activation.pow(2)).unsqueeze(-1) * self.w
        det_term = 1.0 + torch.matmul(psi, self.u)
        log_det = torch.log(det_term.abs() + EPS)
        return z_new, log_det.squeeze(-1)


class NFVRNN(VRNN):
    def __init__(self, *args, flow_layers: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.flows = nn.ModuleList(PlanarFlow(self.latent_dim, device=self.device) for _ in range(flow_layers))

    def _apply_flows(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_sum = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        z_k = z
        for flow in self.flows:
            z_k, log_det = flow(z_k)
            log_det_sum = log_det_sum + log_det
        return z_k, log_det_sum

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

            z0_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t, log_det = self._apply_flows(z0_t)
            phi_z_t = self.phi_z(z_t)

            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], dim=1))
            dec_mean_t = self.dec_mean(dec_t)
            decoded.append(dec_mean_t)

            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(1), h)

            log_q0 = _gaussian_log_prob(z0_t, enc_mean_t, enc_std_t)
            log_p = _gaussian_log_prob(z_t, prior_mean_t, prior_std_t)
            kld_loss = kld_loss + torch.sum(log_q0 - log_p - log_det)

        return kld_loss, torch.stack(decoded, dim=1)


__all__ = [
    "BetaDLGM",
    "BetatDLGM",
    "ConditionalVRNN",
    "IWAEVRNN",
    "LadderVAE",
    "KalmanVAE",
    "SRNN",
    "NFVRNN",
]
