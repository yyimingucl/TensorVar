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

from dataset import ERA5_Dynamics_Dataset, ERA5_DA_Dataset
from train import train_inv_obs_model, train_forward_model
from model.ERA5_model.ERA5_model import ERA5_C_FORWARD, ERA5_C_INVERSE
from utils import dict2namespace


cur_path = os.path.abspath(__file__)
model_save_folder = 'model_weights'

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("[INFO] Using {} device".format(device))
print("[INFO] ", torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU')


# Load the configuration file
config_path = os.path.join('config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
config = dict2namespace(config)

# Load data
data_save_path = '../../data/ERA5_data/train_seq_state.h5'
dynamics_dataset = ERA5_Dynamics_Dataset(data_path=data_save_path, seq_length=config.seq_length)
weight_matrix = np.load("../../data/ERA5_data/weight_matrix.npy")

forward_model = ERA5_C_FORWARD(config)
for i in range(config.num_epochs // config.decay_step):
    batch_size = config.batch_size 
    num_epochs = config.decay_step
    learning_rate = config.learning_rate * config.decay_rate ** i
    print(f"[INFO] Training forward model for {num_epochs} epochs with batch size {batch_size} and learning rate {learning_rate}")
    loss_info = train_forward_model(forward_model, dynamics_dataset, 
                                    batch_size=batch_size, num_epochs=num_epochs, 
                                    learning_rate=learning_rate,
                                    model_save_folder=model_save_folder, 
                                    device=device,
                                    lamb=config.lamb,
                                    weight_matrix=weight_matrix)
forward_model.load_state_dict(torch.load(model_save_folder + '/' + 'forward_model.pt'))
forward_model.C_fwd = torch.load(model_save_folder + '/' + 'C_fwd.pt')
forward_model.to(device)
dynamics_dataset = ERA5_Dynamics_Dataset(data_path=data_save_path, seq_length=1)
forward_model.compute_z_b(dynamics_dataset, device=device, save_path=model_save_folder)
# forward_model.compute_Q_B(dynamics_dataset, device=device, save_path=model_save_folder)
del dynamics_dataset


obs_data_save_path = '../../data/ERA5_data/train_seq_obs.h5'
da_dataset = ERA5_DA_Dataset(state_data_path=data_save_path, 
                             obs_data_path=obs_data_save_path,
                             history_len=config.history_len)
inverse_model = ERA5_C_INVERSE(config)
inverse_model.K_S = forward_model.K_S
inverse_model.K_S_preimage = forward_model.K_S_preimage

for i in range(config.num_epochs // config.decay_step):
    batch_size = config.batch_size 
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
                                    nu=config.nu,
                                    weight_matrix=weight_matrix)

# Compute Q, R, and B
# inverse_model.compute_R(da_dataset, 
#                         device=device, 
#                         save_path=model_save_folder,
#                         weight_matrix=weight_matrix)
