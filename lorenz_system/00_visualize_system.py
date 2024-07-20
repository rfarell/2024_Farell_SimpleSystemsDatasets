import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Assuming ODELorenz and LorenzDataset classes are defined in another file (e.g., my_system.py)
from my_system import ODELorenz, LorenzDataset

def visualize_trajectories(trajectories, t):
    yts = trajectories['yts']
    fig = plt.figure(figsize=(12, 10))

    # Plotting x, y, z trajectories
    ax = fig.add_subplot(221, projection='3d')
    for i, yt in enumerate(yts):
        x, y, z = yt[:, 0], yt[:, 1], yt[:, 2]
        ax.plot(x, y, z, label=f'Trajectory {i+1}')
    ax.set_title('3D Trajectories')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Plotting X vs Time
    ax1 = fig.add_subplot(222)
    for i, yt in enumerate(yts):
        x = yt[:, 0]
        ax1.plot(t, x, label=f'Trajectory {i+1}')
    ax1.set_title('X over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X')
    ax1.legend()
    ax1.grid(True)

    # Plotting Y vs Time
    ax2 = fig.add_subplot(223)
    for i, yt in enumerate(yts):
        y = yt[:, 1]
        ax2.plot(t, y, label=f'Trajectory {i+1}')
    ax2.set_title('Y over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)

    # Plotting Z vs Time
    ax3 = fig.add_subplot(224)
    for i, yt in enumerate(yts):
        z = yt[:, 2]
        ax3.plot(t, z, label=f'Trajectory {i+1}')
    ax3.set_title('Z over Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Z')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('lorenz_trajectories.png')
    plt.show()

def main():
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    T = 10
    timescale = 100
    samples = 5
    x_range = (-20.0, 20.0)
    y_range = (-30.0, 30.0)
    z_range = (0.0, 50.0)
    sigma_noise = 0.1

    lorenz_data = LorenzDataset(sigma, rho, beta, T, timescale, samples, x_range, y_range, z_range, sigma_noise)
    trajectories = lorenz_data.generate_trajectories()

    visualize_trajectories(trajectories, trajectories['t'])

if __name__ == "__main__":
    main()
