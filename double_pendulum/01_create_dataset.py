import sys
import os
import torch
from torchdiffeq import odeint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'systems')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from double_pendulum import DoublePendulumDataset

if __name__ == '__main__':
    mass1 = 1.0  # kg
    mass2 = 0.1  # kg
    gravity = 9.81  # m/s^2
    length1 = 1.0  # m
    length2 = 2.0  # m
    friction = 0.00  # kg/s
    T = 5.0  # s
    timescale = 100  # observations/second
    train_samples = 1000
    test_samples = 100
    q1_lower = 0.1  # rad
    q1_upper = 0.5  # rad
    q2_lower = 0.0  # rad
    q2_upper = 0.0  # rad
    sigma = 0.00  # m

    # Create training dataset
    train_dataset = DoublePendulumDataset(mass1, mass2, length1, length2, gravity, friction, T, timescale, train_samples, q1_lower, q1_upper, q2_lower, q2_upper, sigma)
    train_data = train_dataset.generate_trajectories()
    torch.save(train_data, 'double_pendulum_train_trajectories.pth')

    # Create testing dataset
    test_dataset = DoublePendulumDataset(mass1, mass2, length1, length2, gravity, friction, T, timescale, test_samples, q1_lower, q1_upper, q2_lower, q2_upper, sigma)
    test_data = test_dataset.generate_trajectories()
    torch.save(test_data, 'double_pendulum_test_trajectories.pth')

    # Optionally, load the data again to check
    loaded_train_data = torch.load('double_pendulum_train_trajectories.pth')
    loaded_test_data = torch.load('double_pendulum_test_trajectories.pth')
    print(f'Train data keys: {loaded_train_data.keys()}')
    print(f'Test data keys: {loaded_test_data.keys()}')
