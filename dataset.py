"""
Filename: dataset.py
Author: Yiming Yang (zcahyy1@ucl.ac.uk)
Created: 11/03/2024
Description:
    This file contains the class for Data assimilation dataset for training.
"""
from torch.utils.data import Dataset
import torch
import jax.numpy as jnp
import numpy as np
# from Simulation.SimLorenz96_old import SimLorenz96
import h5py


class DA_Dataset(Dataset):
    def __init__(self, state: np.ndarray, obs: np.ndarray, 
                 hist: np.ndarray, *args, **kwargs):
        super(DA_Dataset, self).__init__(*args, **kwargs)
        """
        :param np.ndarray state_background: [batch_size, state_dim]
        :param np.ndarray obs: [batch_size, obs_dim]
        :param np.ndarray hist: [batch_size, state_dim, history_len]
        """
        if state.ndim==3:
            state = state.reshape(state.shape[0]*state.shape[1], state.shape[2])
        if obs.ndim==3:
            obs = obs.reshape(obs.shape[0]*obs.shape[1], obs.shape[2])
        if hist.ndim==4:
            hist = hist.reshape(hist.shape[0]*hist.shape[1], hist.shape[2], hist.shape[3])
            
        self.state = torch.tensor(state, dtype=torch.float32)
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.hist = torch.tensor(hist, dtype=torch.float32)

    def __len__(self):
        return self.state.shape[0] # -1 for x_t1_background = model(x_t0_analyzed) # No x_(t-1)_analyzed
    
    def __getitem__(self, idx):
        """
        (obs, state_background, hist) => state_analyzed
        """
        return self.obs[idx], self.state[idx], self.hist[idx].permute(1,0) # dim before hist length

class DA_Dynamics_Dataset(Dataset):
    def __init__(self, state: np.ndarray, seq_length=8, *args, **kwargs):
        super(DA_Dynamics_Dataset, self).__init__(*args, **kwargs)
        """
        :param np.ndarray state_background: [batch_size, state_dim]
        """
        if state.ndim==2:
            self.num_traj = 1
            self.num_steps_per_traj, self.state_dim = state.shape
            state = state[np.newaxis, ...]
            print("state shape: ", state.shape)
        elif state.ndim==3:
            self.num_traj, self.num_steps_per_traj, self.state_dim = state.shape
        else:
            raise ValueError("Invalid state shape")
        self.seq_length = seq_length
        self.num_data = self.num_traj * (self.num_steps_per_traj - self.seq_length - 1)
        
        self.create_data_set(state)
    
    def create_data_set(self, state):
        pool = []
        for i in range(self.num_traj):
            for j in range(self.num_steps_per_traj-self.seq_length-1):
                pool.append(state[i, j:j+self.seq_length+1])
        assert len(pool) == self.num_data, "Data size mismatch"
        self.pool = pool
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        """
        (obs, state_background, hist) => state_analyzed
        """
        data = self.pool[idx]
        pre_seq = torch.tensor(data[:self.seq_length], dtype=torch.float32)
        post_seq = torch.tensor(data[1:self.seq_length+1], dtype=torch.float32)
        if self.seq_length == 1:
            pre_seq = pre_seq.squeeze(0)
            post_seq = post_seq.squeeze(0)
        return pre_seq, post_seq

class ERA5_Dynamics_Dataset(Dataset):
    def __init__(self, data_path:str, seq_length:int=8, *args, **kwargs):
        super(ERA5_Dynamics_Dataset, self).__init__(*args, **kwargs)
        """
        :param np.ndarray state_background: [batch_size, state_dim]
        """
        state = h5py.File(data_path, 'r')["data"]
        self.num_steps, self.W, self.H, self.C = state.shape
        
        max_value = np.max(state, axis=(0,1,2))
        min_value = np.min(state, axis=(0,1,2))
        self.max_value = max_value
        self.min_value = min_value
        
        self.seq_length = seq_length
        
        self.num_data = self.num_steps - self.seq_length - 1
        self.create_data_set(state)
    
    def create_data_set(self, state):
        pool = []
        for i in range(self.num_data):
            pool.append(state[i:i+self.seq_length+1])
        assert len(pool) == self.num_data, "Data size mismatch"
        self.pool = pool
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        """
        (obs, state_background, hist) => state_analyzed
        """
        data = self.pool[idx]
        pre_seq = torch.tensor(data[:self.seq_length], dtype=torch.float32).permute(0, 3, 1, 2) 
        post_seq = torch.tensor(data[1:self.seq_length+1], dtype=torch.float32).permute(0, 3, 1, 2) 
        return self.normalize(pre_seq), self.normalize(post_seq)
    
    def normalize(self, x):
        min_value = self.min_value.reshape(1, -1, 1, 1)
        max_value = self.max_value.reshape(1, -1, 1, 1)
        return (x - min_value) / (max_value - min_value)
    
    def denormalizer(self):
        min_value = self.min_value.reshape(1, -1, 1, 1)
        max_value = self.max_value.reshape(1, -1, 1, 1)
        def denormalize(x):
            return x * (max_value - min_value) + min_value
        return denormalize

class ERA5_DA_Dataset(Dataset):
    def __init__(self, state_data_path:str, obs_data_path:str, history_len=5, *args, **kwargs):
        super(ERA5_DA_Dataset, self).__init__(*args, **kwargs)
        """
        :param np.ndarray state_background: [batch_size, state_dim]
        """
        try:
            state = h5py.File(state_data_path, 'r')["test_data"]
        except:
            state = h5py.File(state_data_path, 'r')["data"]
            
        self.num_steps, self.W, self.H, self.C = state.shape
        self.history_len = history_len

        try:
            obs = np.load(obs_data_path)
        except:
            try:
                obs = h5py.File(obs_data_path, 'r')["test_data"]
            except:
                obs = h5py.File(obs_data_path, 'r')["data"]
        
        max_value = np.max(state, axis=(0,1,2))
        min_value = np.min(state, axis=(0,1,2))
        self.max_value = torch.from_numpy(max_value)
        self.min_value = torch.from_numpy(min_value)
        
        self.num_data = self.num_steps - self.history_len
        self.create_data_set(obs, state)
    
    def create_data_set(self, obs, state):
        pool = []
        for i in range(self.history_len, self.num_steps):
            pool.append([obs[i-self.history_len:i], state[i]])
        assert len(pool) == self.num_data, "Data size mismatch"
        self.pool = pool
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        """
        (obs, state_background, hist) => state_analyzed
        """
        data = self.pool[idx]
        obs = torch.tensor(data[0], dtype=torch.float32)
        obs = obs.permute(0, 3, 1, 2).reshape(-1, self.W, self.H)
        state = torch.tensor(data[1], dtype=torch.float32).permute(2, 0, 1)
        return self.normalize(obs), self.normalize(state), 0
    
    def normalize(self, x):
        C = x.shape[0]
        num_rep = x.shape[0] // self.C
        min_value = self.min_value.repeat(num_rep).view(C, 1, 1)
        max_value = self.max_value.repeat(num_rep).view(C, 1, 1)
        return (x - min_value) / (max_value - min_value)
    
    def denormalizer(self):
        def denormalize(x):
            C = x.shape[0]
            num_rep = x.shape[0] // self.C
            min_value = self.min_value.repeat(num_rep).view(C, 1, 1)
            max_value = self.max_value.repeat(num_rep).view(C, 1, 1)
            return x * (max_value - min_value) + min_value
        return denormalize






    



