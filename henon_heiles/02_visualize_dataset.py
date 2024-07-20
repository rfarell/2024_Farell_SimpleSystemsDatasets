import torch
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath)

def plot_trajectories(data, n):
    """Plot the first n trajectories from the dataset."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    for i in range(min(n, len(data['yts']))):  # Ensure we do not exceed available trajectories
        traj = data['yts'][i]
        x, y, px, py = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
        
        axs[0, 0].scatter(x, y, label=f'Trajectory {i+1}', s=10)
        axs[0, 1].scatter(px, py, label=f'Trajectory {i+1}', s=10)
        axs[1, 0].plot(data['t'], x, label=f'Trajectory {i+1}')
        axs[1, 1].plot(data['t'], y, label=f'Trajectory {i+1}')
    
    axs[0, 0].set_title('Phase Space: x vs y')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[0, 1].set_title('Phase Space: px vs py')
    axs[0, 1].set_xlabel('px')
    axs[0, 1].set_ylabel('py')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[1, 0].set_title('x over Time')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('x')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    axs[1, 1].set_title('y over Time')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('henon_heiles_train_trajectories.png')
    plt.show()

if __name__ == '__main__':
    filepath = 'henon_heiles_train_trajectories.pth'  # Path to the data file
    n = 5  # Number of trajectories to plot

    data = load_data(filepath)
    plot_trajectories(data, n)
