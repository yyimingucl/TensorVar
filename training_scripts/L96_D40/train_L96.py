import os
import sys
current_directory = os.getcwd()
upper_directory = os.path.abspath(os.path.join(current_directory, ".."))
upper_upper_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(upper_directory)
sys.path.append(upper_upper_directory)


## ================================================================= ##

import torch
import numpy as np
import yaml

from dataset import DA_Dataset, DA_Dynamics_Dataset
from train import train_inv_obs_model, train_forward_model
from model.Lorenz_model import L96_foward_model, L96_inv_obs_model
from utils import dict2namespace

cur_path = os.path.abspath(__file__)
model_save_folder = 'model_weights'

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("[INFO] Using {} device".format(device))
print("[INFO] ", torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU')

# Load data
data_save_path = '../../data/L96_data_dim40'
seq_state = np.load(data_save_path + '/train_seq_state.npy', mmap_mode='r')
seq_obs = np.load(data_save_path + '/train_seq_obs.npy', mmap_mode='r')
seq_hist = np.load(data_save_path + '/train_seq_hist.npy', mmap_mode='r')

dynamics_dataset = DA_Dynamics_Dataset(state=seq_state)
# Load the configuration file
config_path = os.path.join('config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
config = dict2namespace(config)

forward_model = L96_foward_model(config)
print(forward_model)
for i in range(config.num_epochs // config.decay_step):
    batch_size = config.batch_size * 2**i
    num_epochs = config.decay_step
    learning_rate = config.learning_rate * config.decay_rate ** i
    print(f"[INFO] Training forward model for {num_epochs} epochs with batch size {batch_size} and learning rate {learning_rate}")
    loss_info = train_forward_model(forward_model, dynamics_dataset, 
                                    batch_size=batch_size, num_epochs=num_epochs, 
                                    learning_rate=learning_rate,
                                    model_save_folder=model_save_folder, 
                                    device=device,
                                    lamb=config.lamb)
forward_model.load_state_dict(torch.load(model_save_folder + '/' + 'forward_model.pt'))
forward_model.C_fwd = torch.load(model_save_folder + '/' + 'C_fwd.pt')
forward_model.to(device)
dynamics_dataset = DA_Dynamics_Dataset(state=seq_state, seq_length=1)
forward_model.compute_Q_B(dynamics_dataset, device=device, save_path=model_save_folder)
del dynamics_dataset

da_dataset = DA_Dataset(state=seq_state, obs=seq_obs, hist=seq_hist)
inverse_model = L96_inv_obs_model(config)
inverse_model.phi_S = forward_model.phi_S
inverse_model.phi_inv_S = forward_model.phi_inv_S
print(inverse_model)

for i in range(config.num_epochs // config.decay_step):
    batch_size = config.batch_size * 2**i 
    num_epochs = config.decay_step
    learning_rate = config.learning_rate * config.decay_rate ** i
    print(f"[INFO] Training inverse model for {num_epochs} epochs with batch size {batch_size} and learning rate {learning_rate}")
    loss_info = train_inv_obs_model(inverse_model,
                                    da_dataset, 
                                    batch_size=batch_size, 
                                    num_epochs=num_epochs, 
                                    learning_rate=learning_rate, 
                                    model_save_folder=model_save_folder, 
                                    device=device,
                                    nu=config.nu)

# Compute Q, R, and B
inverse_model.compute_R(da_dataset, 
                        device=device, 
                        save_path=model_save_folder)
