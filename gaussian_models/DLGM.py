import torch.nn as nn
import torch
from collections.abc import Iterator
from itertools import chain


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ── Recognition ───────────────────────────────────────────────────────────────


class RecLayer(nn.Module):

    
    def __init__(self, input_dim=1, latent_dim=1, device=None):
        super().__init__()
        self.latent_dim=latent_dim
        self.input_dim=input_dim
        self.device = device

        self.d = nn.Sequential(
                nn.Linear(self.input_dim, self.latent_dim),
                nn.Sigmoid(),
                nn.Linear(self.latent_dim,self.latent_dim),
                nn.Sigmoid()).to(device)
        self.u = nn.Sequential(
                    nn.Linear(self.input_dim, self.latent_dim),
                    nn.Sigmoid(),
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.Sigmoid()
                ).to(device)
        self.mean = nn.Sequential(
                    nn.Linear(self.input_dim, self.latent_dim),
                    nn.Tanh(),
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.Tanh()
                ).to(device)

    def forward(self, x):
        d = self.d(x)
        u = self.u(x)
        mean = self.mean(x)
        R = self.calculate_r(d,u)
        z = self.calculate_z(mean,R)
        return mean, R, z

    
    def calculate_z(self, mean, R):
        v = torch.randn(*mean.size(), 1, device=self.device)
        mult = torch.matmul(R, v).squeeze(-1)
        return mult + mean
    

    def calculate_r(self, d, u):
        epsilon = 1e-6
        d_safe = d.clamp(min=epsilon)
        D_inv = torch.diag_embed(1.0 / d_safe)
        D_inv_sqrt = torch.sqrt(D_inv)
        u_r = u.unsqueeze(-1)
        U = torch.matmul(u_r, u_r.transpose(-2,-1))
        ut_d_inv_u = torch.matmul(u_r.transpose(-2,-1), torch.matmul(D_inv, u_r))
        eta = 1.0 / (1.0 + ut_d_inv_u)
        right = (1.0 - torch.sqrt(eta)) / (ut_d_inv_u + epsilon)
        R = D_inv_sqrt - right * torch.matmul(D_inv, torch.matmul(U, D_inv_sqrt))
        return R

class Recognition(nn.Module):


    def __init__(self, input_dim=1, latent_dim=1, layers=1, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers=layers+1
        self.device = device

        self.make_network()


    def make_network(self):

        self.g = nn.ModuleList()
        for i in range(self.layers):
            self.g.append(RecLayer(self.input_dim, self.latent_dim, self.device).to(self.device))


    def forward(self, x):
        R = []
        mean = []
        z = []
        for l in self.g:
            res = l(x)
            R.append(res[1])
            mean.append(res[0])
            z.append(res[2])
        
        return mean, R, z


# ── Generator ─────────────────────────────────────────────────────────────────────



class GenLayer(nn.Module):

    def __init__(self, hidden_size=1, latent_dim=1, seq_len=1, device=None):
        super().__init__()
        self.hidden_size=hidden_size
        self.latent_dim=latent_dim
        self.seq_len=seq_len
        self.device=device
        self.t = nn.Sequential(
            nn.Linear(in_features=self.hidden_size,
                            out_features=self.hidden_size,device=device),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.hidden_size, device=device),
            nn.ReLU()
        ).to(self.device)
            
        
        self.g = nn.Sequential(
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.latent_dim,
                            device=self.device),
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.hidden_size,
                            device=self.device),
            nn.LeakyReLU()).to(self.device)

    # Adding the noise and previous layer
    def forward(self, h, xi):
        h = self.t(h)
        return h + self.g(xi)

class Generator(nn.Module):

    def __init__(self, hidden_size=1, latent_dim=1, output_dim=1, layers=1, seq_len=1, device=None):
        super().__init__()
        self.output_dim = output_dim
        self.layers=layers
        self.hidden_size = hidden_size
        self.latent_dim=latent_dim
        self.seq_len=seq_len
        self.device=device
        self.make_network()
        self.xi = None

    def make_network(self):

        self.h_l = nn.ModuleList()

        for i in range(self.layers):
            self.h_l.append(GenLayer(self.hidden_size, self.latent_dim,self.seq_len, self.device))

        self.H_L = nn.Sequential(
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.latent_dim,
                            device=self.device),
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.hidden_size,
                            device=self.device),
            nn.Tanh()).to(self.device)

        self.h_0 = nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.output_dim, device=self.device),
                nn.Sigmoid()).to(self.device)

    def forward(self, batch_size=1):
        if self.xi is None:
            self.make_xi(batch_size)

        v = self.H_L(self.xi[0])
        count = 1
        for h in self.h_l:
            v = h(v, self.xi[count])
            count += 1
        return self.h_0(v)

    def set_xi(self, xi):
        self.xi = xi

    def make_xi(self, batch_size=1):
        self.xi = []
        for i in range(self.layers + 1):
            self.xi.append(torch.normal(mean=torch.zeros(batch_size, self.seq_len, self.latent_dim).to(self.device), std=1)
                           .to(self.device))



# ── DLGM ─────────────────────────────────────────────────────────────────



class DLGM(nn.Module):
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

        self.model_g = Generator(hidden_size,
                                 latent_dim,
                                 output_dim,
                                 layers,
                                 seq_len,
                                 device)

        self.model_r = Recognition(input_dim, 
                                   latent_dim,
                                   layers,
                                   device)

        self.mse = nn.MSELoss()

    def get_parameters(self) -> Iterator[nn.Parameter]:
        return chain(
            self.model_g.parameters(),
            self.model_r.parameters(),
        )

    def _loss(self, y, y_hat, mean, R) -> torch.Tensor:
        epsilon = 1e-6
        target = y.reshape_as(y_hat)
        loss = self.mse(y_hat, target)
        matrix_size = mean[0].size(0) * mean[0].size(1)

        for m, r in zip(mean, R, strict=False):
            C = r @ r.transpose(-2, -1)
            eye = torch.eye(C.size(-1), device=C.device, dtype=C.dtype).expand_as(C)
            _, logdet = torch.linalg.slogdet(C + epsilon * eye)
            loss += (
                0.5
                * torch.sum(
                    m.pow(2).sum(-1)
                    + C.diagonal(dim1=-2, dim2=-1).sum(-1)
                    - logdet
                    - 1
                )
                / matrix_size
            )
        return loss

    def get_loss(self, x, x_1, y) -> float:
        with torch.no_grad():
            return self.train_step(x, x_1, y, optimizer=None)

    def train_step(self, x, x_1, y, optimizer) -> float:
        if optimizer is not None:
            optimizer.zero_grad()

        mean, R, z = self.model_r(x_1)
        self.model_g.set_xi(z)

        pred = self.model_g(x.size(0))[:,-1,:].unsqueeze(1)

        loss = self._loss(y, pred, mean, R)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
        return loss.item()

    def forward(self, x) -> torch.Tensor:
        rec = self.model_r(x)
        self.model_g.set_xi(rec[2])
        val = self.model_g(x.size(0))
        return val



# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.optim import Adam

    model = DLGM(
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
    y = torch.randn(50, 1, 10).to(device)  ## the value to be reconstructed
    x_1 = torch.cat((x, y), dim=1)[:, 1:, :]  ## used for the recognition

    before = model.get_loss(x, x_1, y)
    for _ in range(300):
        loss = model.train_step(x, x_1, y, optimizer)
    after = model.get_loss(x, x_1, y)
    print(f"Loss before training: {before}")
    print(f"Loss after training: {after}")
    assert after < before, "Loss did not decrease after training! MSELoss"
