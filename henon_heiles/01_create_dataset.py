from my_system import HenonHeilesDataset
import torch

if __name__ == '__main__':
    T = 1.  # s
    timescale = 50  # observations/second
    train_samples = 100
    test_samples = 100
    x_lower = -0.5  # initial condition range for x
    x_upper = 0.5   # initial condition range for x
    y_lower = -0.5  # initial condition range for y
    y_upper = 0.5   # initial condition range for y
    sigma = 0.01  # noise level

    # Create training dataset
    train_dataset = HenonHeilesDataset(T, timescale, train_samples, x_lower, x_upper, y_lower, y_upper, sigma)
    train_data = train_dataset.generate_trajectories()
    torch.save(train_data, 'henon_heiles_train_trajectories.pth')

    # Create testing dataset
    test_dataset = HenonHeilesDataset(T, timescale, test_samples, x_lower, x_upper, y_lower, y_upper, sigma)
    test_data = test_dataset.generate_trajectories()
    torch.save(test_data, 'henon_heiles_test_trajectories.pth')

    # Optionally, load the data again to check
    loaded_train_data = torch.load('henon_heiles_train_trajectories.pth')
    loaded_test_data = torch.load('henon_heiles_test_trajectories.pth')
    print(f'Train data keys: {loaded_train_data.keys()}')
    print(f'Test data keys: {loaded_test_data.keys()}')
