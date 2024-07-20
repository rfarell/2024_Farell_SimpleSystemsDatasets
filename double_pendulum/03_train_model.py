import torch
from collections import defaultdict
from torchdiffeq import odeint
import time
import copy
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'systems')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from ssgp import SSGP

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath)

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If best_model, saves separately"""
    torch.save(state, filename)

def get_batch(x, t_eval, batch_step):
    n_samples, n_points, input_dim = x.shape
    N = n_samples

    # Using torch to generate indices
    n_ids = torch.arange(N)  # equivalent to np.arange(N)
    # Randomly select starting points for each trajectory
    p_ids = torch.randint(0, n_points - batch_step, (N,))  # replace np.random.choice

    batch_x0 = x[n_ids, p_ids].reshape([N, 1, input_dim])
    batch_step += 1
    batch_t = t_eval[:batch_step]
    batch_x = torch.stack([x[n_ids, p_ids + i] for i in range(batch_step)], dim=0).reshape([batch_step, N, 1, input_dim])

    return batch_x0, batch_t, batch_x

def arrange(x, t_eval):
    n_samples, n_points, input_dim = x.shape

    # Using torch to generate indices
    n_ids = torch.arange(n_samples)  # equivalent to np.arange
    p_ids = torch.zeros(n_samples, dtype=torch.int64)  # replace np.array with zero-initialized tensor

    batch_x0 = x[n_ids, p_ids].reshape([n_samples, 1, input_dim])
    batch_t = t_eval
    batch_x = torch.stack([x[n_ids, p_ids + i] for i in range(n_points)], dim=0).reshape([n_points, n_samples, 1, input_dim])

    return batch_x0, batch_t, batch_x

def train(model, train_data, val_data, learning_rate, batch_time, total_steps):
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    stats = defaultdict(list)
    min_val_loss = 1e+10
    t_eval = train_data['t'].clone().detach()
    batch_step = int(((len(t_eval)-1) / t_eval[-1]).item() * batch_time)
    
    for step in range(total_steps+1):
        # train step
        batch_y0, batch_t, batch_ys = get_batch(train_data['yts'], t_eval, batch_step)
        s_batch_x0 = model.sampling_x0(batch_y0)
        model.sampling_epsilon_f()
        # pred_x = odeint(model, s_batch_x0, batch_t, method='dopri5', atol=1e-8, rtol=1e-8)
        pred_x = odeint(model, s_batch_x0, batch_t, method='fehlberg2', atol=1e-4, rtol=1e-4)
        
        neg_loglike = model.neg_loglike(batch_ys, pred_x)
        KL_x0 = model.KL_x0(batch_y0.squeeze())
        KL_w = model.KL_w()
        loss = neg_loglike + KL_w #+ KL_x0
        loss.backward(); optim.step(); optim.zero_grad()
        train_loss = loss.detach().item() / batch_y0.shape[0] / batch_t.shape[0]
        
        # run validation data
        with torch.no_grad():
            batch_y0, batch_t, batch_ys = arrange(val_data['yts'], t_eval)
            s_batch_x0 = model.sampling_x0(batch_y0)
            model.mean_w()
            # pred_val_x = odeint(model, s_batch_x0, t_eval, method='dopri5', atol=1e-8, rtol=1e-8)
        
            pred_val_x = odeint(model, s_batch_x0, t_eval, method='fehlberg2', atol=1e-4, rtol=1e-4)
            val_neg_loglike = model.neg_loglike(batch_ys, pred_val_x)
            val_Kl_x0 = model.KL_x0(batch_y0.squeeze())
            val_Kl_w = model.KL_w()
            val_loss = val_neg_loglike + val_Kl_w + val_Kl_x0
            val_loss = val_neg_loglike.item() / batch_y0.shape[0] / t_eval.shape[0]
        # logging
        stats['train_loss'].append(train_loss)
        stats['train_kl_x0'].append(KL_x0.item())
        stats['train_kl_w'].append(KL_w.item())
        stats['train_neg_loglike'].append(neg_loglike.item() / batch_y0.shape[0] / batch_t.shape[0])
        stats['val_loss'].append(val_loss)
        stats['val_kl_x0'].append(val_Kl_x0.item())
        stats['val_kl_w'].append(val_Kl_w.item())
        stats['val_neg_loglike'].append(val_neg_loglike.item() / batch_y0.shape[0] / t_eval.shape[0])
        if step % 100 == 0:
            print(f"step {step}, train_loss {train_loss:.4e}, val_loss {val_loss:.4e}")

        if val_loss < min_val_loss:
            best_model = copy.deepcopy(model)
            min_val_loss = val_loss; best_train_loss = train_loss
            best_step = step
            # save it
            save_checkpoint({
                'step': step,
                'state_dict': model.state_dict(),
                'optim_dict': optim.state_dict(),
                'stats': stats,
                'best_train_loss': best_train_loss,
                'min_val_loss': min_val_loss,
                'best_step': best_step
            }, filename='best.pth.tar')
            generate_trajectory_plots(best_model, train_data,i=0)
            
    return best_model, optim, stats, best_train_loss, min_val_loss, best_step

def generate_trajectory_plots(model, val_data, i, filename_prefix='trajectory_plot'):
    """
    Generate trajectory plots for the double pendulum using the given model.
    """
    t_eval = val_data['t'].clone().detach()
    batch_step = int(((len(t_eval)-1) / t_eval[-1]).item() * batch_time)
    
    # run validation data
    with torch.no_grad():
        batch_y0, batch_t, batch_ys = arrange(val_data['yts'], t_eval)
        s_batch_x0 = model.sampling_x0(batch_y0)
        model.mean_w()
        # pred_val_x = odeint(model, s_batch_x0, t_eval, method='dopri5', atol=1e-8, rtol=1e-8)
    
        pred_val_x = odeint(model, s_batch_x0, t_eval, method='fehlberg2', atol=1e-4, rtol=1e-4)
    plt.figure(figsize=(15, 7))
    
    # Create subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    for i in range(i,i+1):  # Plot first 5 trajectories
        true_traj = batch_ys[:,i,0,:]
        pred_traj = pred_val_x[:, i, 0, :]
        
        # True trajectories
        theta1_true = true_traj[:, 0]
        theta2_true = true_traj[:, 1]
        p_theta1_true = true_traj[:, 2]
        p_theta2_true = true_traj[:, 3]
        
        # Predicted trajectories
        theta1_pred = pred_traj[:, 0]
        theta2_pred = pred_traj[:, 1]
        p_theta1_pred = pred_traj[:, 2]
        p_theta2_pred = pred_traj[:, 3]

        # True
        ax1.plot(theta1_true, p_theta1_true, label=f'True Trajectory {i+1}', linestyle='-')
        ax2.plot(theta2_true, p_theta2_true, label=f'True Trajectory {i+1}', linestyle='-')
        
        # Predicted
        ax1.plot(theta1_pred, p_theta1_pred, label=f'Predicted Trajectory {i+1}', linestyle='--')
        ax2.plot(theta2_pred, p_theta2_pred, label=f'Predicted Trajectory {i+1}', linestyle='--')
    
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
    
    plt.suptitle('True and Predicted Trajectories')
    plt.savefig(f'{filename_prefix}.png')
    plt.show()

if __name__ == "__main__":
    # Load train and validation data
    train_data = load_data('double_pendulum_train_trajectories.pth')
    test_data = load_data('double_pendulum_test_trajectories.pth')

    input_dim = 4
    num_basis = 100
    friction = True
    K = 100
    learning_rate = 1e-3
    batch_time = 2
    total_steps = 10000

    # Initialize the model
    model = SSGP(input_dim, num_basis, friction, K)


    # Learning
    t0 = time.time()
    best_model, optim, stats, train_loss, val_loss, step = train(model, train_data, test_data, learning_rate, batch_time, total_steps)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f} s")

    # Generate trajectory plots
    generate_trajectory_plots(best_model, train_data,i=0)

    # pred_x.shape, batch_ys.shape
    # import matplotlib.pyplot as plt
    # i = 50
    # plt.scatter(pred_x[:,i,0,0].detach().numpy(), pred_x[:,i,0,1].detach().numpy())
    # plt.scatter(batch_ys[:,i,0,0].detach().numpy(), batch_ys[:,i,0,1].detach().numpy())
    
    # i = 10
    # plt.scatter(pred_x[:,i,0,2].detach().numpy(), pred_x[:,i,0,3].detach().numpy())
    # plt.scatter(batch_ys[:,i,0,2].detach().numpy(), batch_ys[:,i,0,3].detach().numpy())