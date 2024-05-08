import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 40, 2)  # Correctly calculate the number of input features
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2, 8 * 32 * 40),
            nn.ReLU(),
            nn.Unflatten(1, (8, 32, 40)),
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
    
    def forward(self, x):
        return self.decoder(x)


class HamiltonianNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 20)  # Input to hidden
        self.fc2 = nn.Linear(20, 20)  # Hidden to hidden
        self.fc3 = nn.Linear(20, 1)   # Hidden to output (Hamiltonian)
        self.activation = nn.Tanh()

    def forward(self, qp):
        h1 = self.activation(self.fc1(qp))
        h2 = self.activation(self.fc2(h1))
        H = self.fc3(h2)  # Hamiltonian
        return H

def dynamics(t, qp, hamiltonian_model):
    qp = qp.requires_grad_(True)
    H = hamiltonian_model(qp)
    dH_dqp = torch.autograd.grad(H.sum(), qp, create_graph=True)[0]

    dq_dt = dH_dqp[:, 1]
    dp_dt = -dH_dqp[:, 0]

    return torch.cat([dq_dt.unsqueeze(1), dp_dt.unsqueeze(1)], dim=1)

def predict_final_state(hamiltonian_model, qp0, T_scalar):
    if torch.is_tensor(T_scalar):
        T_scalar = T_scalar.item()
    t = torch.tensor([0, T_scalar], dtype=qp0.dtype, device=qp0.device)
    trajectory = odeint(lambda t, y: dynamics(t, y, hamiltonian_model), qp0, t, method='dopri5', atol=1e-3, rtol=1e-3)
    return trajectory[-1]

def predict_full_trajectory(hamiltonian_model, qp0, T_scalar, t_points=10):
    # Ensure T_scalar is a scalar by extracting its item if it's a tensor
    if torch.is_tensor(T_scalar):
        T_scalar = T_scalar.item()
    # Generate time points from 0 to T
    t = torch.linspace(0, T_scalar, t_points, dtype=qp0.dtype, device=qp0.device)
    # Compute the trajectory using the ODE solver
    trajectory = odeint(lambda t, y: dynamics(t, y, hamiltonian_model), qp0, t, method='dopri5', atol=1e-3, rtol=1e-3)
    return trajectory  # Return the full trajectory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
hamiltonian_model = HamiltonianNN().to(device)
print('Encoder Parameters:', sum(p.numel() for p in encoder.parameters()))
print('Decoder Parameters:', sum(p.numel() for p in decoder.parameters()))
print('Hamiltonian Model Parameters:', sum(p.numel() for p in hamiltonian_model.parameters()))


criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(hamiltonian_model.parameters()), lr=0.001)

# Load the data
data = np.load('train.npy')
data = data[:, None, :, :]  # Add channel dimension
data = torch.tensor(data, dtype=torch.float32)

N = data.shape[0]
S = 3
T = 10
K = 100

# Preallocate the array for inputs
data_Input = torch.zeros((N - S + 1, S, 128, 160))

# Populate data_Input with each set of three consecutive frames
for i in range(N - S + 1):
    data_Input[i] = data[i:i+S].reshape(S, 128, 160)  # Reshape to remove the singleton dimension

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    i = np.random.randint(0, N - T - K-1)
    batch = data_Input[i:i+T+K]
    batch.shape
    # encode the batch
    latent = encoder(batch)
    qp_predictions = []
    for j in range(K):
        qp0 = latent[j, :2].unsqueeze(0)
        qp_pred = predict_full_trajectory(hamiltonian_model, qp0, T_scalar=T, t_points=T+1)
        qp_predictions.append(qp_pred)
    qp_predictions = torch.cat(qp_predictions, dim=0).squeeze(1)
    pred_outputs = decoder(qp_predictions)

    loss_output = 0
    for k in range(0, K):
        for t in range(0,T+1):
            loss_output += criterion(pred_outputs[k*(T+1) + t], data[S+i+k+t])
    
    loss_latent = 0
    for k in range(0, K):
        for t in range(0,T+1):
            loss_latent += criterion(qp_predictions[k*(T+1)+t], latent[k+t])

    # Compute loss metween pred_outputs and batch
    loss = loss_output/(K*(T+1)) + loss_latent/(K*(T+1))
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Output Loss: {loss_output.item():.4f}, Latent Loss: {loss_latent.item():.4f}')

# Parameters for dataset creation
max_t = 10  # Number of output frames after the input
total_frames = data.shape[0]

# Initialize lists to store the expanded dataset
expanded_X = []
expanded_Y = []

# Loop through the dataset to generate data points
for i in range(total_frames - 2 - max_t):
    # Append the input frames
    expanded_X.append(data[i:i+3])  # Three consecutive frames
    # Append the next 10 frames as the target
    expanded_Y.append(data[i+3:i+3+max_t])  # Next 10 frames

# Convert the lists to numpy arrays
X_array = np.stack(expanded_X)
X_array = np.squeeze(X_array, axis=2)

Y_array = np.stack(expanded_Y)
Y_array = np.squeeze(Y_array, axis=2)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_array, dtype=torch.float32)
Y_tensor = torch.tensor(Y_array, dtype=torch.float32)

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
hamiltonian_model = HamiltonianNN().to(device)
print('Encoder Parameters:', sum(p.numel() for p in encoder.parameters()))
print('Decoder Parameters:', sum(p.numel() for p in decoder.parameters()))
print('Hamiltonian Model Parameters:', sum(p.numel() for p in hamiltonian_model.parameters()))


criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(hamiltonian_model.parameters()), lr=0.001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    for inputs,  targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)        

        # Forward pass
        latent = encoder(inputs)
        # target_latent = encoder(targets)
        qp_predictions = []
        for i in range(latent.shape[0]):
            qp0 = latent[i, :2].unsqueeze(0)
            T_scalar = 10  # Normalize t to be between 0 and 1
            qp_pred = predict_full_trajectory(hamiltonian_model, qp0, T_scalar, t_points=10)
            qp_predictions.append(qp_pred)
        qp_predictions = torch.cat(qp_predictions, dim=0)
        outputs = decoder(qp_predictions.squeeze(1))
        
        # Compute loss
        output_loss = criterion(outputs, targets)

        # Add MSE loss between qp_predictions and target_latent
        latent_loss = criterion(qp_predictions, target_latent)

        # loss
        loss = output_loss + latent_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Output Loss: {output_loss.item():.4f}, Latent Loss: {latent_loss.item():.4f}')

 
# save models
torch.save(encoder.state_dict(), 'encoder.pth')
torch.save(decoder.state_dict(), 'decoder.pth')
torch.save(hamiltonian_model.state_dict(), 'hamiltonian_model.pth')

# read in models
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load('encoder.pth'))
encoder.eval()

decoder = Decoder().to(device)
decoder.load_state_dict(torch.load('decoder.pth'))
decoder.eval()

hamiltonian_model = HamiltonianNN().to(device)
hamiltonian_model.load_state_dict(torch.load('hamiltonian_model.pth'))
hamiltonian_model.eval()


# Print parameter coutns
print('Encoder Parameters:', sum(p.numel() for p in encoder.parameters()))
print('Decoder Parameters:', sum(p.numel() for p in decoder.parameters()))
print('Hamiltonian Model Parameters:', sum(p.numel() for p in hamiltonian_model.parameters()))


for inputs, t, full_targets, targets in dataloader:
    inputs, t, full_targets, targets = inputs.to(device), t.to(device), full_targets.to(device), targets.to(device)        
    # Forward pass
    latent = encoder(inputs)
    qp_predictions = []
    for i in range(latent.shape[0]):
        qp0 = latent[i, :2].unsqueeze(0)
        T_scalar = t[i]
        qp_pred = predict_final_state(hamiltonian_model, qp0, T_scalar)
        qp_predictions.append(qp_pred)
    qp_predictions = torch.cat(qp_predictions, dim=0)
    
    predicted = decoder(qp_predictions)

    # Move tensors to CPU for plotting
    inputs = inputs.cpu()
    predicted = predicted.cpu()
    targets = targets.cpu()

    for i in range(3):
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        
        # Plot each of the three input frames
        for j in range(3):
            ax = axes[j]
            ax.imshow(inputs[i, j].squeeze(), cmap='gray')
            ax.set_title(f'Input Frame {j+1}')
            ax.axis('off')
        
        # Plot the predicted frame
        axes[3].imshow(predicted[i].squeeze().detach().numpy(), cmap='gray')
        axes[3].set_title('Predicted Frame')
        axes[3].axis('off')

        # Plot the actual frame
        axes[4].imshow(targets[i].squeeze(), cmap='gray')
        axes[4].set_title('Actual Frame')
        axes[4].axis('off')
        
        plt.show()

    break  # Only process one batch


import cv2

for inputs, t, full_targets, targets in dataloader:
    inputs, t, full_targets, targets = inputs.to(device), t.to(device), full_targets.to(device), targets.to(device)        

    # Forward pass
    latent = encoder(inputs)
    qp0 = latent[0, :2].unsqueeze(0)
    T_scalar = 50
    qp_pred = predict_full_trajectory(hamiltonian_model, qp0, T_scalar, t_points=100)
    qp_pred = qp_pred.squeeze(1)
    predicted = decoder(qp_pred)
    
    # Move tensors to CPU for plotting
    inputs = inputs.cpu()
    predicted = predicted.cpu().detach()
    break

def tensor_to_video_opencv(tensor, filename='output.avi', fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    height, width = tensor.shape[2], tensor.shape[3]
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=False)

    for i in range(tensor.size(0)):
        frame = tensor[i, 0].numpy()
        frame = np.uint8((frame / frame.max()) * 255)
        out.write(frame)

    out.release()

# Call the function
tensor_to_video_opencv(predicted.detach())

