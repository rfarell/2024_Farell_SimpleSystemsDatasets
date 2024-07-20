import torch
import matplotlib.pyplot as plt
from cycler import cycler

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath)

def plot_trajectories(data, n):
    """Plot the first n trajectories from the dataset."""
    plt.figure(figsize=(15, 7))
    
    # Define a color cycle
    colors = plt.cm.tab10.colors
    color_cycle = cycler(color=colors)
    
    # Create subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    ax1.set_prop_cycle(color_cycle)
    ax2.set_prop_cycle(color_cycle)
    
    for i in range(min(n, len(data['yts']))):  # Ensure we do not exceed available trajectories
        traj = data['yts'][i]
        theta1 = traj[:, 0]
        theta2 = traj[:, 1]
        p_theta1 = traj[:, 2]
        p_theta2 = traj[:, 3]
        
        ax1.scatter(theta1, p_theta1, label=f'Trajectory {i+1}', s=10)
        ax2.scatter(theta2, p_theta2, label=f'Trajectory {i+1}', s=10)
    
    # Configure subplot for Theta1 vs p_theta1
    ax1.set_title('Theta1 vs p_theta1')
    ax1.set_xlabel('Theta1 (rad)')
    ax1.set_ylabel('p_theta1')
    ax1.legend()
    ax1.grid(True)
    
    # Configure subplot for Theta2 vs p_theta2
    ax2.set_title('Theta2 vs p_theta2')
    ax2.set_xlabel('Theta2 (rad)')
    ax2.set_ylabel('p_theta2')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f'First {n} Trajectories out of {len(data["yts"])} Trajectories')
    plt.savefig('double_pendulum_train_trajectories.png')
    plt.show()

if __name__ == '__main__':
    filepath = 'double_pendulum_train_trajectories.pth'  # Path to the data file
    n = 10  # Number of trajectories to plot

    data = load_data(filepath)
    plot_trajectories(data, n)
