from my_system import LorenzDataset
import torch

if __name__ == '__main__':
    sigma = 10.0  # Lorenz system parameter
    rho = 28.0    # Lorenz system parameter
    beta = 8.0 / 3.0  # Lorenz system parameter
    T = 10.0  # s
    timescale = 100  # observations/second
    train_samples = 100
    test_samples = 100
    x_range = (-20.0, 20.0)  # range for initial x
    y_range = (-30.0, 30.0)  # range for initial y
    z_range = (0.0, 50.0)    # range for initial z
    sigma_noise = 0.1  # noise level

    # Create training dataset
    train_dataset = LorenzDataset(sigma, rho, beta, T, timescale, train_samples, x_range, y_range, z_range, sigma_noise)
    train_data = train_dataset.generate_trajectories()
    torch.save(train_data, 'lorenz_train_trajectories.pth')

    # Create testing dataset
    test_dataset = LorenzDataset(sigma, rho, beta, T, timescale, test_samples, x_range, y_range, z_range, sigma_noise)
    test_data = test_dataset.generate_trajectories()
    torch.save(test_data, 'lorenz_test_trajectories.pth')

    # Optionally, load the data again to check
    loaded_train_data = torch.load('lorenz_train_trajectories.pth')
    loaded_test_data = torch.load('lorenz_test_trajectories.pth')
    print(f'Train data keys: {loaded_train_data.keys()}')
    print(f'Test data keys: {loaded_test_data.keys()}')
