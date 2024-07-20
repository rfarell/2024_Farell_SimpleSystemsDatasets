import torch
import matplotlib.pyplot as plt
import sys
import os
from torchdiffeq import odeint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'systems')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from ssgp import SSGP

def load_checkpoint(filepath):
    """
    Load the model checkpoint from the given file path.
    """
    checkpoint = torch.load(filepath, map_location='cpu')  # Ensure it loads on CPU
    return checkpoint

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath)

def plot_losses(stats):
    """
    Plot the training and validation losses from the training statistics.
    """
    train_losses = stats['train_loss']
    val_losses = stats['val_loss']
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('losses.png')
    plt.close()

def plot_phase_space(predicted, actual, n=5):
    """
    Plot the phase space of the predicted and actual trajectories.
    
    Parameters:
    - predicted (torch.Tensor): Predicted trajectories.
    - actual (torch.Tensor): Actual trajectories.
    - n (int): Number of trajectories to plot.
    """
    fig, axes = plt.subplots(n, 2, figsize=(15, 5 * n))
    
    for i in range(min(n, len(actual))):
        pred_traj = predicted[i]
        actual_traj = actual[i]
        
        theta1_pred = pred_traj[:, 0].detach().numpy()
        theta2_pred = pred_traj[:, 1].detach().numpy()
        p_theta1_pred = pred_traj[:, 2].detach().numpy()
        p_theta2_pred = pred_traj[:, 3].detach().numpy()
        
        theta1_actual = actual_traj[:, 0].detach().numpy()
        theta2_actual = actual_traj[:, 1].detach().numpy()
        p_theta1_actual = actual_traj[:, 2].detach().numpy()
        p_theta2_actual = actual_traj[:, 3].detach().numpy()
        
        # Theta1 vs p_theta1
        axes[i, 0].plot(theta1_actual, p_theta1_actual, label='Actual', linestyle='--')
        axes[i, 0].plot(theta1_pred, p_theta1_pred, label='Predicted', linestyle='-')
        axes[i, 0].set_title(f'Sample {i+1}: Theta1 vs p_theta1')
        axes[i, 0].set_xlabel('Theta1 (rad)')
        axes[i, 0].set_ylabel('p_theta1')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Theta2 vs p_theta2
        axes[i, 1].plot(theta2_actual, p_theta2_actual, label='Actual', linestyle='--')
        axes[i, 1].plot(theta2_pred, p_theta2_pred, label='Predicted', linestyle='-')
        axes[i, 1].set_title(f'Sample {i+1}: Theta2 vs p_theta2')
        axes[i, 1].set_xlabel('Theta2 (rad)')
        axes[i, 1].set_ylabel('p_theta2')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('double_pendulum_phase_space.png')
    plt.close()

if __name__ == '__main__':
    model_filepath = 'best.pth.tar'  # Path to the saved model checkpoint
    test_data_filepath = 'double_pendulum_test_trajectories.pth'  # Path to the test data

    checkpoint = load_checkpoint(model_filepath)
    
    # Assuming 'stats' is a dictionary containing lists of loss values
    stats = checkpoint['stats']
    plot_losses(stats)

    input_dim = 4
    num_basis = 100
    friction = True
    K = 100
    learning_rate = 1e-3
    batch_time = 1
    total_steps = 2000

    # Initialize the model
    model = SSGP(input_dim, num_basis, friction, K)

    # Load the model parameters
    model.load_state_dict(checkpoint['state_dict'])
    model.sampling_epsilon_f()

    # Load the test data
    test_data = load_data(test_data_filepath)
    actual_trajectories = test_data['yts']
    t = test_data['t']
    
    # Generate predictions
    initial_conditions = actual_trajectories[:, 0, :].clone().detach()
    # initial_conditions is of shape torch.Size([100, 4]) but should be shape torch.Size([100, 1, 4])
    initial_conditions = initial_conditions.unsqueeze(1)
    predicted_trajectories = odeint(model, initial_conditions, t, method='dopri5', atol=1e-8, rtol=1e-8)
    # predicted_trajectories is of shape  torch.Size([101, 100, 1, 4]) but should be shape torch.Size([101, 100, 4])
    predicted_trajectories = predicted_trajectories.squeeze(2)
    predicted_trajectories = predicted_trajectories.permute(1, 0, 2)

    # torch.Size([100, 1, 4]) torch.Size([101])    
    # Plot the phase space of the predicted vs actual trajectories
    plot_phase_space(predicted_trajectories, actual_trajectories, n=5)