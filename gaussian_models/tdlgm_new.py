from itertools import chain

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Time Recognition ──────────────────────────────────────────────────────────


class TimeLayer(nn.Module):
    def __init__(self, input_dim=1, hidden_size=1, device=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            device=self.device,
        )

    def forward(self, x):
        _, h = self.lstm(x)
        return h


class TimeRecognition(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        seq_len=1,
        layers=1,
        device=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.layers = layers
        self.device = device
        self.input_dim = input_dim

        self.time_layers = nn.ModuleList(
            TimeLayer(
                input_dim=input_dim,
                hidden_size=hidden_size,
                device=device,
            )
            for _ in range(layers)
        )

    def forward(self, x):
        return [layer(x) for layer in self.time_layers]


# ── Generator ─────────────────────────────────────────────────────────────────


class GenLayer(nn.Module):
    def __init__(
        self,
        hidden_size=1,
        latent_dim=1,
        seq_len=1,
        device=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.device = device

        self.internal_state = None

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            device=self.device,
        )

        self.g = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                self.latent_dim,
                device=self.device,
            ),
            nn.Linear(
                self.latent_dim,
                self.hidden_size,
                device=self.device,
            ),
            nn.LeakyReLU(),
        )

    def get_internal_state(self):
        return self.internal_state

    def set_internal_state(self, internal_state):
        if internal_state is None:
            self.internal_state = None
            return

        # Prevent carrying computation graphs between sequences
        self.internal_state = (
            internal_state[0].detach(),
            internal_state[1].detach(),
        )

    def make_internal_state(self, batch_size=1):
        self.internal_state = (
            torch.zeros(
                1,
                batch_size,
                self.hidden_size,
                device=self.device,
            ),
            torch.zeros(
                1,
                batch_size,
                self.hidden_size,
                device=self.device,
            ),
        )

    def forward(self, h, xi):
        h, self.internal_state = self.lstm(
            h,
            self.internal_state,
        )

        # Corresponds to:
        # h_l,t = R_l(h_(l+1), s_l,t) + G_l xi_l,t
        return h + self.g(xi)


class Generator(nn.Module):
    def __init__(
        self,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        device=None,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.layers = layers
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.device = device

        self.xi = None

        self.gen_layers = nn.ModuleList(
            GenLayer(
                hidden_size,
                latent_dim,
                seq_len,
                device,
            )
            for _ in range(layers)
        )

        self.initial_transform = nn.Sequential(
            nn.Linear(
                latent_dim,
                latent_dim,
                device=device,
            ),
            nn.Linear(
                latent_dim,
                hidden_size,
                device=device,
            ),
            nn.Tanh(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(
                hidden_size,
                output_dim,
                device=device,
            ),
            nn.Sigmoid(),
        )

    def forward(self, batch_size=1):

        if self.xi is None:
            self.make_xi(batch_size)

        v = self.initial_transform(self.xi[0])

        for i, layer in enumerate(
            self.gen_layers,
            start=1,
        ):
            v = layer(
                v,
                self.xi[i],
            )

        return (
            self.output_layer(v[:, -1, :]),
            self.get_internal_state(),
        )

    def get_internal_state(self):
        return [layer.get_internal_state() for layer in self.gen_layers]

    def set_internal_state(self, internal_state):
        for layer, state in zip(
            self.gen_layers,
            internal_state,
            strict=False,
        ):
            layer.set_internal_state(state)

    def make_internal_state(self, batch_size=1):
        for layer in self.gen_layers:
            layer.make_internal_state(batch_size)

    def set_xi(self, xi):
        self.xi = xi

    def make_xi(self, batch_size=1):
        self.xi = [
            torch.randn(
                batch_size,
                self.seq_len,
                self.latent_dim,
                device=self.device,
            )
            for _ in range(self.layers + 1)
        ]
        # ── Recognition ───────────────────────────────────────────────────────────────


class RecLayer(nn.Module):
    def __init__(
        self,
        input_dim=1,
        latent_dim=1,
        device=None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

        self.d = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Sigmoid(),
            nn.Linear(latent_dim, latent_dim),
            nn.Softplus(),
        ).to(device)

        self.u = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Sigmoid(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        ).to(device)

        self.mean = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        ).to(device)

    def forward(self, x):

        mean = self.mean(x)

        # d represents diagonal covariance contribution
        d = self.d(x)

        # u represents low-rank contribution
        u = self.u(x)

        R = self._calculate_r(
            d,
            u,
        )

        z = self._sample(
            mean,
            R,
        )

        return mean, R, z

    def _sample(
        self,
        mean,
        R,
    ):
        eps = torch.randn_like(mean)

        return mean + torch.matmul(
            R,
            eps.unsqueeze(-1),
        ).squeeze(-1)

    def _calculate_r(
        self,
        d,
        u,
    ):
        """
        Stable factorization of:

        C = D + uu^T

        where:

        R R^T = C

        """

        eps = 1e-6

        d = d + eps

        D_inv = torch.diag_embed(1.0 / d)

        D_inv_sqrt = torch.diag_embed(torch.sqrt(1.0 / d))

        u_col = u.unsqueeze(-1)

        uu = torch.matmul(
            u_col,
            u_col.transpose(-2, -1),
        )

        denom = 1.0 + torch.matmul(
            u_col.transpose(-2, -1),
            torch.matmul(
                D_inv,
                u_col,
            ),
        )

        coefficient = (1.0 - torch.sqrt(1.0 / denom)) / (denom + eps)

        R = D_inv_sqrt - coefficient * torch.matmul(
            D_inv,
            torch.matmul(
                uu,
                D_inv_sqrt,
            ),
        )

        return R


class Recognition(nn.Module):
    def __init__(
        self,
        input_dim=1,
        latent_dim=1,
        layers=1,
        device=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers = layers
        self.device = device

        self.rec_layers = nn.ModuleList(
            RecLayer(
                input_dim,
                latent_dim,
                device,
            )
            for _ in range(layers + 1)
        )

    def forward(self, x):

        means = []
        Rs = []
        zs = []

        for layer in self.rec_layers:
            mean, R, z = layer(x)

            means.append(mean)
            Rs.append(R)
            zs.append(z)

        return means, Rs, zs


# ── tDLGM ─────────────────────────────────────────────────────────────────────


class tDLGM(nn.Module):
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

        self.model_t = TimeRecognition(
            input_dim,
            hidden_size,
            seq_len,
            layers,
            device,
        )

        self.model_g = Generator(
            hidden_size,
            latent_dim,
            output_dim,
            layers,
            seq_len,
            device,
        )

        self.model_r = Recognition(
            input_dim,
            latent_dim,
            layers,
            device,
        )

        self.mse = nn.MSELoss()

    def get_parameters(self):

        return chain(
            self.model_t.parameters(),
            self.model_g.parameters(),
            self.model_r.parameters(),
        )

    def gaussian_kl(
        self,
        mean,
        R,
    ):
        """
        KL(q(z)||N(0,I))

        q(z)=N(mean,C)

        C=RR^T

        """

        C = torch.matmul(
            R,
            R.transpose(-2, -1),
        )

        trace = C.diagonal(
            dim1=-2,
            dim2=-1,
        ).sum(-1)

        logdet = torch.linalg.slogdet(C)[1]

        return 0.5 * torch.sum(mean.pow(2).sum(-1) + trace - logdet - mean.size(-1))

    def _loss(
        self,
        y,
        y_hat,
        mean,
        R,
        s,
        t_1,
        reg,
    ):

        loss = self.mse(
            y_hat,
            y.reshape_as(y_hat),
        )

        # KL latent loss
        kl = 0

        for m, r in zip(
            mean,
            R,
            strict=False,
        ):
            kl += self.gaussian_kl(
                m,
                r,
            )

        loss += kl / y.size(0)

        # State consistency regularization
        state_loss = 0

        amount = len(s) * len(s[0])

        for a, b in zip(
            s,
            t_1,
            strict=False,
        ):
            state_loss += self.mse(
                a[0],
                b[0],
            ) + self.mse(
                a[1],
                b[1],
            )

        loss += reg * state_loss / amount

        return loss

    def get_loss(
        self,
        x,
        x_1,
        y,
    ):

        return self.train_step(
            x,
            x_1,
            y,
            optimizer=None,
        )

    def train_step(
        self,
        x,
        x_1,
        y,
        optimizer,
    ):

        if optimizer is not None:
            optimizer.zero_grad()

        # State recognition
        t = self.model_t(x)

        t_1 = self.model_t(x_1)

        # Reset recurrent states
        self.model_g.make_internal_state(x.size(0))

        self.model_g.set_internal_state(t)

        # Latent recognition
        mean, R, z = self.model_r(x_1)

        self.model_g.set_xi(z)

        prediction, h = self.model_g(x.size(0))

        loss = self._loss(
            y,
            prediction,
            mean,
            R,
            h,
            t_1,
            reg=0.01,
        )

        if optimizer is not None:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        return loss.item()

    def forward(
        self,
        x,
    ):

        self.model_g.make_internal_state(x.size(0))

        t = self.model_t(x)

        self.model_g.set_internal_state(t)

        self.model_g.make_xi(x.size(0))

        value, _ = self.model_g(x.size(0))

        return value


# ── Self Test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from torch.optim import Adam

    model = tDLGM(
        input_dim=10,
        hidden_size=20,
        latent_dim=5,
        output_dim=10,
        layers=2,
        seq_len=3,
        device=device,
    ).to(device)

    optimizer = Adam(
        model.get_parameters(),
        lr=1e-3,
    )

    x = torch.randn(
        50,
        3,
        10,
    ).to(device)

    y = torch.randn(
        50,
        1,
        10,
    ).to(device)

    x_1 = torch.cat(
        (
            x,
            y,
        ),
        dim=1,
    )[:, 1:, :]

    before = model.get_loss(
        x,
        x_1,
        y,
    )

    for _ in range(300):
        model.train_step(
            x,
            x_1,
            y,
            optimizer,
        )

    after = model.get_loss(
        x,
        x_1,
        y,
    )

    print(f"Loss before training: {before}")

    print(f"Loss after training: {after}")

    assert after < before


    assert after < before
