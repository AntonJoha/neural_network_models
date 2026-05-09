import torch
import torch.nn as nn

devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# ── VRNNN ─────────────────────────────────────────────────────────────────────

class VRNNN(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        activation_function=nn.ReLU,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layers = layers
        self.activation_function = activation_function


        self.phi_prior = nn.Linear(hidden_size,
                               latent_dim * 2)

        self.phi_encoder = nn.Linear(input_dim + hidden_size,
                                 latent_dim * 2)

        self.phi_decoder = nn.Linear(latent_dim + hidden_size,
                                 output_dim)

        self.phi_x = nn.Linear(input_dim, hidden_size)

        self.phi_z = nn.Linear(latent_dim, hidden_size)

        self.rnn = nn.GRU(input_dim + latent_dim,
                          hidden_size,
                          layers)



  
    def _reparameterize(self, mean, var_diag):
        eps = torch.randn_like(var_diag)
        z = mean + eps * var_diag
        return z

    def forward(self, x, h_t_1):
        prior = self.prior(h_t_1)

        mean_prior, var_prior = prior[:, :self.latent_dim], prior[:, self.latent_dim:]

        z_t = self._reparameterize(mean_prior, var_prior)
        
        phi_z_t = self.phi_z(z_t)
        phi_x_t = self.phi_x(x)

        x_decoded = self.phi_decoder(torch.cat([phi_z_t, h_t_1], dim=1))

        posterior = self.phi_encoder(torch.cat([phi_x_t, h_t_1], dim=1))
        mean_post, var_post = posterior[:, :self.latent_dim], posterior[:, self.latent_dim:]
        z_post = self._reparameterize(mean_post, var_post)



