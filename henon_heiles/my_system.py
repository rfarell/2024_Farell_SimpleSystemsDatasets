import torch
from torch import nn
from torchdiffeq import odeint

class ODEHenonHeiles(nn.Module):
    def __init__(self):
        super().__init__()
        # Skew-symmetric matrix for Hamiltonian mechanics
        self.S = torch.tensor([[0, 1, 0, 0], 
                               [-1, 0, 0, 0], 
                               [0, 0, 0, 1], 
                               [0, 0, -1, 0]]).float()

    def hamiltonian(self, coords):
        x, y, px, py = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        H = 0.5 * (px.pow(2) + py.pow(2)) + 0.5 * (x.pow(2) + y.pow(2)) + x.pow(2) * y - (1/3) * y.pow(3)
        return H
    
    def forward(self, t, x):
        H = self.hamiltonian(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        field = dH @ self.S.t()
        return field

class HenonHeilesDataset:
    def __init__(self, T, timescale, samples, x_lower, x_upper, y_lower, y_upper, sigma):
        self.ode = ODEHenonHeiles()
        self.samples = samples
        self.T = T
        self.timescale = timescale
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.y_lower = y_lower
        self.y_upper = y_upper
        self.sigma = sigma

    def get_initial_conditions(self):
        x0s = (self.x_upper - self.x_lower) * torch.rand(self.samples, 1) + self.x_lower
        y0s = (self.y_upper - self.y_lower) * torch.rand(self.samples, 1) + self.y_lower
        px0s = torch.zeros(self.samples, 1)
        py0s = torch.zeros(self.samples, 1)
        initial_conditions = torch.cat([x0s, y0s, px0s, py0s], dim=1)
        initial_conditions.requires_grad_(True)
        return initial_conditions

    def generate_trajectories(self):
        x0s = self.get_initial_conditions()
        t = torch.linspace(0, self.T, int(self.T * self.timescale + 1))
        xts = odeint(self.ode, x0s, t, method='dopri5', atol=1e-6, rtol=1e-6)  # Adjust tolerances
        noise = torch.randn(xts.shape) * self.sigma
        yts = xts + noise
        yts = yts.permute(1, 0, 2)
        return {'yts': yts.detach(), 't': t.detach()}

# # Example usage
# henon_heiles_data = HenonHeilesDataset(
#     T=10,
#     timescale=50,  # Adjust timescale
#     samples=10,
#     x_lower=-0.5,  # Adjust initial condition range
#     x_upper=0.5,
#     y_lower=-0.5,
#     y_upper=0.5,
#     sigma=0.01
# )

# trajectories = henon_heiles_data.generate_trajectories()
# print(trajectories)
