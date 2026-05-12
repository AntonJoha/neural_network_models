import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DEVICE = device
EPS = 1e-6


def epsilon_for(tensor: torch.Tensor) -> float:
    return max(EPS, torch.finfo(tensor.dtype).eps)


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
        activation_function=nn.ReLU,
        reconstruction_loss_factory=nn.MSELoss,
    ):
        """Create a VRNN.

        activation_function should be a callable that returns an nn.Module
        (for example nn.ReLU, nn.LeakyReLU).
        """
        super().__init__()
        if device is None:
            device = DEFAULT_DEVICE
        device = torch.device(device)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layers = layers
        self.seq_len = seq_len
        self.device = device
        self.activation_function = activation_function
        self.reconstruction_loss_factory = reconstruction_loss_factory

        self.phi_x = nn.Sequential(
            nn.Linear(input_dim, hidden_size, device=device),
            activation_function(),
        )
        self.phi_z = nn.Sequential(
            nn.Linear(latent_dim, hidden_size, device=device),
            activation_function(),
        )

        self.enc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size, device=device),
            activation_function(),
        )
        self.enc_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_size, latent_dim, device=device),
            nn.Softplus(),
        )

        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, device=device),
            activation_function(),
        )
        self.prior_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_size, latent_dim, device=device),
            nn.Softplus(),
        )

        self.dec = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size, device=device),
            activation_function(),
        )
        self.dec_mean = nn.Linear(hidden_size, output_dim, device=device)

        self.rnn = nn.GRU(
            input_size=hidden_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            device=device,
        )

        self.reconstruction_loss = reconstruction_loss_factory().to(device)

    def get_parameters(self):
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
        epsilon = epsilon_for(std_post)
        post_var = std_post.pow(2).clamp(min=epsilon)
        prior_var = std_prior.pow(2).clamp(min=epsilon)
        kld = (
            torch.log(prior_var)
            - torch.log(post_var)
            + (post_var + (mean_post - mean_prior).pow(2)) / prior_var
            - 1.0
        )
        return 0.5 * torch.sum(kld)

    def _validate_sequence_input(self, x: torch.Tensor, name: str, expected_dim: int) -> None:
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(f"{name} must be a 3D tensor [batch, seq, features], got shape {tuple(x.shape)}")
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError(f"{name} must have non-zero batch and sequence dimensions, got shape {tuple(x.shape)}")
        if x.size(2) != expected_dim:
            raise ValueError(
                f"{name} feature dimension must be {expected_dim}, got {x.size(2)}"
            )
        if x.device != self.device:
            raise ValueError(
                f"{name} device {x.device} does not match model device {self.device}"
            )

    def _forward_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_sequence_input(x, "x", self.input_dim)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(self.layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        kld_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        decoded = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            phi_x_t = self.phi_x(x_t)

            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], dim=1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) + epsilon_for(enc_t)

            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t) + epsilon_for(prior_t)

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
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        if y.numel() != y_hat.numel():
            raise ValueError(f"Target shape {tuple(y.shape)} is incompatible with prediction shape {tuple(y_hat.shape)}")
        target = y.reshape_as(y_hat)
        recon = self.reconstruction_loss(y_hat, target)
        kld = kld_loss / (batch_size * seq_len)
        return recon + kld

    def get_loss(self, _x, x_1, y) -> float:
        with torch.no_grad():
            return self._compute_loss(x_1, y).item()

    def _compute_loss(self, x_1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._validate_sequence_input(x_1, "x_1", self.input_dim)
        self._validate_sequence_input(y, "y", self.output_dim)
        kld_loss, decoded = self._forward_sequence(x_1)
        pred = decoded[:, -1, :].unsqueeze(1)
        return self._loss(y, pred, kld_loss, x_1.size(0), x_1.size(1))

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


# Backward-compatible alias for the old typo'd class name.
VRNNN = VRNN
