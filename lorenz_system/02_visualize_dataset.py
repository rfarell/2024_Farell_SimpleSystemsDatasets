import torch
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath)

def plot_trajectories(data, n):
    """Plot the first n trajectories from the dataset."""
    yts = data['yts']
    t = data['t']
    fig = plt.figure(figsize=(12, 10))

    # Plotting x, y, z trajectories
    ax = fig.add_subplot(221, projection='3d')
    for i, yt in enumerate(yts[:n]):  # Ensure we do not exceed available trajectories
        x, y, z = yt[:, 0], yt[:, 1], yt[:, 2]
        ax.plot(x, y, z, label=f'Trajectory {i+1}')
    ax.set_title('3D Trajectories')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Plotting X vs Time
    ax1 = fig.add_subplot(222)
    for i, yt in enumerate(yts[:n]):
        x = yt[:, 0]
        ax1.plot(t, x, label=f'Trajectory {i+1}')
    ax1.set_title('X over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X')
    ax1.legend()
    ax1.grid(True)

    # Plotting Y vs Time
    ax2 = fig.add_subplot(223)
    for i, yt in enumerate(yts[:n]):
        y = yt[:, 1]
        ax2.plot(t, y, label=f'Trajectory {i+1}')
    ax2.set_title('Y over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)

    # Plotting Z vs Time
    ax3 = fig.add_subplot(224)
    for i, yt in enumerate(yts[:n]):
        z = yt[:, 2]
        ax3.plot(t, z, label=f'Trajectory {i+1}')
    ax3.set_title('Z over Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Z')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('lorenz_train_trajectories.png')
    plt.show()

if __name__ == '__main__':
    filepath = 'lorenz_train_trajectories.pth'  # Path to the data file
    n = 5  # Number of trajectories to plot

    data = load_data(filepath)
    plot_trajectories(data, n)
