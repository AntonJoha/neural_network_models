# TODO tDLGM: Temporal Deep Latent Gaussian Model

from itertools import chain

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################
########################Time layer#####################
########################################################

class TimeLayer(nn.Module):


    def __init__(self, input_dim=1, hidden_size=1, seq_len=1, device=None):
        super().__init__()
            
        self.input_dim=input_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.device = device

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_size,num_layers=1, batch_first=True).to(self.device)
    
    def init_hidden(self, size):
        self.internal_state = [
                    torch.zeros(1, size[0], self.hidden_size,device=self.device).to(self.device),
                    torch.zeros(1, size[0], self.hidden_size,device=self.device).to(self.device)
                ]



    def forward(self, x):
        self.init_hidden(x.size())
        _, h = self.lstm(x)# , self.internal_state)
        return h

class TimeRecognition(nn.Module):



    def __init__(self, input_dim=1, hidden_size=1, seq_len=1, layers=1, device=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.layers = layers
        self.device = device
        self.input_dim=input_dim

        self.make_network()

    def forward(self, x):
        res = []
        for l in self.h:
            res.append(l(x))
        return res
    
    def make_network(self):

        self.h = nn.ModuleList()

        for _i in range(self.layers):
            self.h.append(TimeLayer(input_dim=self.input_dim,
                                    hidden_size=self.hidden_size,
                                    seq_len=self.seq_len,
                                    device=self.device).to(self.device))



########################################################
########################Generator#######################
########################################################


class GenLayer(nn.Module):

    def __init__(self, hidden_size=1, latent_dim=1, seq_len=1, device=None):
        super().__init__()
        self.hidden_size=hidden_size
        self.latent_dim=latent_dim
        self.seq_len=seq_len
        self.device=device

        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True,
                            device=self.device)
        
        self.g = nn.Sequential(
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.latent_dim,
                            device=self.device),
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.hidden_size,
                            device=self.device),
            nn.LeakyReLU()).to(self.device)


    def get_internal_state(self):
        return self.internal_state

    # Adding the noise and previous layer
    def forward(self, h, xi):
        h, self.internal_state = self.lstm(h, self.internal_state)
        return h + self.g(xi)

    def set_internal_state(self, internal_state):
        self.internal_state=internal_state
    
    def make_internal_state(self, batch_size=1):
        self.internal_state = [
                    torch.zeros(1, batch_size, self.hidden_size).to(self.device),
                    torch.zeros(1, batch_size, self.hidden_size).to(self.device)
                ]

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

        for _i in range(self.layers):
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
        return self.h_0(v[:,-1,:]), self.get_internal_state()


    def get_internal_state(self):
        states = []
        for layer in self.h_l:
            states.append(layer.get_internal_state())
        return states

    def set_internal_state(self, internal_state):
        for layer, state in zip(self.h_l, internal_state, strict=False):
            layer.set_internal_state(state)

    def make_internal_state(self, batch_size=1):
        for layer in self.h_l:
            layer.make_internal_state(batch_size)

    def set_xi(self, xi):
        self.xi = xi

    def make_xi(self, batch_size=1):
        self.xi = []
        for _i in range(self.layers + 1):
            self.xi.append(torch.normal(mean=torch.zeros(batch_size, self.seq_len, self.latent_dim).to(self.device), std=1)
                           .to(self.device))


########################################################
########################Recognition#######################
########################################################

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
        v = torch.randn(mean.size()).unsqueeze(-1).to(self.device)
        mult = torch.matmul(R, v).squeeze()
        return mult + mean
    

    def calculate_r(self, d, u):
        D = torch.diag_embed(d)
        epsilon = 1e-6
        D_inv = torch.inverse(D)  + epsilon
        D_in_sqr = torch.sqrt(D_inv)
        u_r = u.unsqueeze(-1)
        U = torch.matmul(u_r, u_r.transpose(-2,-1))
        ut_d_inv_u = torch.matmul(u_r.transpose(-2,-1), torch.matmul(D_inv, u_r))
        eta = 1/(1 + ut_d_inv_u)
        right = (1 - torch.sqrt(eta)) / ut_d_inv_u
        R = D_in_sqr - right*torch.matmul(D_inv, torch.matmul(U, D_in_sqr))
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
        for _i in range(self.layers):
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



########################################################
##########TIME DEEP LATENT GAUSSIAN MODEL###############
########################################################



class tDLGM(nn.Module):

    def __init__(self, input_dim=1, hidden_size=1, latent_dim=1, output_dim=1, layers=1, seq_len=1, device=None):
        super().__init__()

        self.model_t = TimeRecognition(input_dim, hidden_size, seq_len, layers, device)
        self.model_g = Generator(hidden_size, latent_dim, output_dim, layers, seq_len, device)
        self.model_r = Recognition(input_dim, latent_dim, layers, device)
        
        self.mse = nn.MSELoss()


    
    def get_parameters(self)-> chain[nn.Parameter]:
        return chain(self.model_t.parameters(), self.model_g.parameters(), self.model_r.parameters())



    def _loss(self, y, y_hat, mean, R, s ,x_1, reg, was, seq_len)-> torch.Tensor:

        l = self.mse(y_hat, y.squeeze()) # Why is it sum here? Should it be mean? Do not remember TODO: Check this
        matrix_size = mean[0].size()[0]*mean[0].size()[1]

        for m, r in zip(mean, R, strict=False):
            C = r @ r.transpose(-2,-1) 
            det = C.det()
            l += 0.5* torch.sum(m.pow(2).sum(-1) + C.diagonal(dim1=-2, dim2=-1).sum(-1) - det.log() - 1)/matrix_size


        amount = len(s)*len(s[0])
        for a, b in zip(s, x_1, strict=False):
            l += reg*(self.mse(a[0], b[0]) + self.mse(a[1], b[1]))/amount

        return l

    
    def get_loss(self, x, x_1, y) -> float:
        return self.train_step(x, x_1, y, optimizer=None)



    def train_step(self, x, x_1, y, optimizer) -> float:

        if optimizer is not None:
            optimizer.zero_grad()

        t = self.model_t(x)
        t_1 = self.model_t(x_1)
        self.model_g.make_internal_state(x.size(0))
        self.model_g.set_internal_state(t)
        rec = self.model_r(x_1)
        self.model_g.set_xi(rec[2])

        pred, h = self.model_g(x.size(0))

        loss = self._loss(y,
                          pred,
                          rec[0],
                          rec[1],
                          h,
                          t_1,
                          reg=0.01,
                          was=False,
                          seq_len=self.model_g.seq_len)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
        return loss.item()


    def forward(self, x) -> torch.Tensor:
        self.model_g.make_internal_state(x.size(0))
        t = self.model_t(x)
        self.model_g.set_internal_state(t)
        self.model_g.make_xi(x.size(0))
        val, _ = self.model_g(x.size(0))
        return val
        





########################################################
################TEST TEST TEST #########################
########################################################

if __name__ == "__main__":
    from torch.optim import Adam

    model = tDLGM(input_dim=10, hidden_size=20, latent_dim=5, output_dim=10, layers=2, seq_len=3, device=device).to(device)
    optimizer = Adam(model.get_parameters(), lr=0.1)

    x = torch.randn(50,3,10).to(device) ## used for the state recognition
    y = torch.randn(50,1,10).to(device) ## the value to be reconstructed
    x_1 = torch.cat((x,y), dim=1)[:,1:,:] ## used for the recognition

    before = model.get_loss(x,x_1,y)
    for _i in range(300):
        loss = model.train_step(x, x_1, y, optimizer)
    after = model.get_loss(x,x_1,y)
    print(f"Loss before training: {before}")
    print(f"Loss after training: {after}")
    assert after < before, "Loss did not decrease after training!"
    

