import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
import torch
from torchdiffeq import odeint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'systems')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from double_pendulum import DoublePendulum


def plot_double_pendulum_motion():
    """
    Animates the double pendulum motion using the positions from the trajectory data alongside a phase-space plot.

    Parameters:
    - xts (torch.Tensor): Tensor containing the trajectory data with shape [num_steps, 1, 4]
                            where each entry is [theta1, theta2, p_theta1, p_theta2].
    - t (torch.Tensor): Tensor containing the time steps.
    - fps (int): Frames per second for the animation (default 100).
    """
    # Parameters for the double pendulum
    mass1 = 10.0  # kg
    mass2 = 1.0  # kg
    gravity = 9.81  # m/s^2
    length1 = 1.0  # m
    length2 = 1.0  # m
    friction = 0.01  # kg/s

    ode = DoublePendulum(mass1, mass2, length1, length2, gravity, friction)

    # Visualize the double pendulum motion
    qp0 = torch.tensor([[0.5, 0.0, 0.0, 0.0]], requires_grad=True)
    t = torch.linspace(0, 10, 101)
    xts = odeint(ode, qp0, t, method='dopri5', atol=1e-8, rtol=1e-8)
    theta1 = xts[:, 0, 0].detach().numpy()
    theta2 = xts[:, 0, 1].detach().numpy()
    p_theta1 = xts[:, 0, 2].detach().numpy()
    p_theta2 = xts[:, 0, 3].detach().numpy()
    times = t.numpy()

    # Initialize the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.set_title('Double Pendulum Motion')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    line, = ax1.plot([], [], 'o-', lw=2, markersize=8)  # Line object to update in the animation for double pendulum

    # Set up the phase space plot
    ax2.set_title('Phase Space')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-np.max(np.abs(p_theta1))*1.1, np.max(np.abs(p_theta1))*1.1)
    ax2.set_xlabel('$q_1$')
    ax2.set_ylabel('$p_1$')
    line_phase, = ax2.plot([], [], 'r-', alpha=0.5)  # Line object for phase space trajectory

    def init():
        """Initialize the background of both plots."""
        line.set_data([], [])
        line_phase.set_data([], [])
        return line, line_phase

    def update(frame):
        """Updates the position of the pendulum and phase space for each frame."""
        x1 = length1 * np.sin(theta1[frame])
        y1 = -length1 * np.cos(theta1[frame])
        x2 = x1 + length2 * np.sin(theta2[frame])
        y2 = y1 - length2 * np.cos(theta2[frame])
        line.set_data([0, x1, x2], [0, y1, y2])

        line_phase.set_data(theta1[:frame + 1], p_theta1[:frame + 1])
        return line, line_phase

    ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True, interval=50, repeat=False)

    # Save to file
    ani.save('./double_pendulum_animation.mp4', writer='ffmpeg', dpi=300)

    # Close the plot
    plt.close(fig)

if __name__ == '__main__':
    plot_double_pendulum_motion()
