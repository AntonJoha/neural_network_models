from collections.abc import Iterator
from itertools import chain

import torch
import torch.nn as nn
from .tDLGM import tDLGM, device


# ── Time Recognition ──────────────────────────────────────────────────────────


class TimeLayer(nn.Module):


    def __init__(self, batch_dim=1, input_dim=1, state_dim=1,network_size=[1], activation_function=nn.LeakyReLU, device=None):
        super().__init__()

        self.batch_dim = batch_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.activation_function = activation_function
        self.device = device
        self.network_input = self.input_dim*self.batch_dim
        self.network_size = network_size
        self.make_network()

    def make_network(self):


        network = [self.network_input]
        for i in self.network_size:
            network.append(i)

        layers = []
        for i in range(len(network)-1):
            layers.append(nn.Linear(in_features=network[i],out_features=network[i+1]))
            layers.append(self.activation_function())
        layers.append(nn.Linear(in_features=network[-1], out_features=self.state_dim))

        self.mean = nn.Sequential(*layers)
        self.var = nn.Sequential(*layers)
    

    def forward(self, x):
        log_var = self.var(x)
        mean = self.mean(x)
        return mean, log_var, mean + self.reparameterization(log_var)


    def reparameterization(self,log_var):
        dims = log_var.size()
        eps = Variable(torch.FloatTensor(dims).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std)


class TimeRecognition(nn.Module):


    def __init__(self, input_dim:int=1, network_size=[1], batches=[1],seq_len=1, state_dim=1):
        super().__init__()

        
        self.state_dim = state_dim
        self.network_size = network_size
        self.batches = batches
        self.input_dim = input_dim
        self.seq_len=seq_len

        self.network = []
        self.make_network()

    def make_network(self):
        self.network = []
        for b in self.batches:
            self.network.append(TimeLayer(batch_dim=b, input_dim=self.input_dim,state_dim=self.state_dim,network_size=self.network_size))


    def get_data(self, seq, batches):
        p = list(torch.split(seq, batches, dim=1)[-self.seq_len:])
        #padding
        entries = p[0].shape[1]
        if p[-1].shape[1] != entries:
            p[-1] = torch.cat((p[-1], torch.zeros(p[-1].shape[0], entries - p[-1].shape[1], p[-1].shape[2])), dim=1)

        res = torch.stack(p, dim=1) # Shape: (batch_size, seq_len, time_steps, features)
        

        res = res.reshape(res.shape[0], res.shape[1], -1) # Shape: (batch_size, seq_len, time_steps*features)
        return res


    def forward(self, x):
        res = []
        mean = []
        r = []
        z = []
        for b, n in zip(self.batches, self.network):
            m, r_curr, z_curr = n(self.get_data(x,b))
            mean.append(m)
            r.append(r_curr)
            z.append(z_curr)
        

        return mean, r, z

# ── Generator ─────────────────────────────────────────────────────────────────

### Can use the same for now. 


# ── tDLGM ─────────────────────────────────────────────────────────────────────

class tDLGM_attention(tDLGM):

    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        batch
        device=None,
        ):
        super().__init__(input_dim, 
                         hidden_size,
                         latent_dim,
                         output_dim,
                         layers,
                         seq_len,
                         device)


        # Override the model_t within the TimeRecognition module, see the constructor of tDLGM for more details.
        self.model_t = TimeRecognition(input_dim=input_dim,
                                                network_size=network_size,
                                                batches=batches,
                                                seq_len=seq_len,
                                                state_dim=state_dim,
                                                device=device)



if __name__ == "__main__":
    from torch.optim import Adam

    model = tDLGM_attention(
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


