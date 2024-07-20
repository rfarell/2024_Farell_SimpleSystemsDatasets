import torch
from torch import nn
from torchdiffeq import odeint

class ODELorenz(nn.Module):
    def __init__(self, sigma, rho, beta):
        super().__init__()
        # Lorenz system parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def forward(self, t, state):
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.stack([dx, dy, dz], dim=1)

class LorenzDataset:
    def __init__(self, sigma, rho, beta, T, timescale, samples, x_range, y_range, z_range, sigma_noise):
        self.ode = ODELorenz(sigma, rho, beta)
        self.samples = samples
        self.T = T
        self.timescale = timescale
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.sigma_noise = sigma_noise

    def get_initial_conditions(self):
        x0s = (self.x_range[1] - self.x_range[0]) * torch.rand(self.samples, 1) + self.x_range[0]
        y0s = (self.y_range[1] - self.y_range[0]) * torch.rand(self.samples, 1) + self.y_range[0]
        z0s = (self.z_range[1] - self.z_range[0]) * torch.rand(self.samples, 1) + self.z_range[0]
        initial_conditions = torch.cat([x0s, y0s, z0s], dim=1)
        initial_conditions.requires_grad_(True)
        return initial_conditions

    def generate_trajectories(self):
        x0s = self.get_initial_conditions()
        t = torch.linspace(0, self.T, int(self.T * self.timescale + 1))
        xts = odeint(self.ode, x0s, t, method='dopri5', atol=1e-8, rtol=1e-8)
        noise = torch.randn(xts.shape) * self.sigma_noise
        yts = xts + noise
        yts = yts.permute(1, 0, 2)
        return {'yts': yts.detach(), 't': t.detach()}
