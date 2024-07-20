import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import numpy as np

from my_system import ODEHenonHeiles, HenonHeilesDataset

def visualize_trajectories(trajectories, t):
    yts = trajectories['yts']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    for yt in yts:
        x, y, px, py = yt[:, 0], yt[:, 1], yt[:, 2], yt[:, 3]
        
        axs[0, 0].plot(x, y)
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')
        axs[0, 0].set_title('Phase Space: x vs y')
        
        axs[0, 1].plot(px, py)
        axs[0, 1].set_xlabel('px')
        axs[0, 1].set_ylabel('py')
        axs[0, 1].set_title('Phase Space: px vs py')
        
        axs[1, 0].plot(t, x)
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('x')
        axs[1, 0].set_title('x over Time')
        
        axs[1, 1].plot(t, y)
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('y')
        axs[1, 1].set_title('y over Time')

    plt.tight_layout()
    plt.show()

def main():
    T = 1
    timescale = 50
    samples = 10
    x_lower = -0.5  # Single value
    x_upper = 0.5   # Single value
    y_lower = -0.5  # Single value
    y_upper = 0.5   # Single value
    sigma = 0.01

    henon_heiles_data = HenonHeilesDataset(T, timescale, samples, x_lower, x_upper, y_lower, y_upper, sigma)
    trajectories = henon_heiles_data.generate_trajectories()
    
    visualize_trajectories(trajectories, trajectories['t'])

if __name__ == "__main__":
    main()
