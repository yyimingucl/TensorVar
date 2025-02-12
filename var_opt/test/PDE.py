import os 
import sys
home_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(home_dir)

import numpy as np
import tqdm
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit
from jax import random as jax_random

from var_opt.LBFGS_Var import Var_Opt_PDE
from KS_solver import create_forward_model

if __name__ == "__main__":
    n_state = 128
    n_obs = int(n_state * 0.25)

    ass_w = 5 # assimilation window 
    ass_T = ass_w # assimilation horizon 

    forward_model = create_forward_model(n_state=n_state)
    obs_index = np.linspace(0, n_state, n_obs, dtype=int)
    obs_index[-1] = n_state - 1
    obs_index = jnp.array(obs_index) # type: ignore

    n_obs = len(obs_index)

    # observation model
    @jit
    def obs_transformation(x):
        key = jax_random.PRNGKey(2024)
        return 5 * jnp.arctan(x[obs_index] * 0.1) + 0.1 * jax_random.normal(key, shape=x[obs_index].shape)
    
    # Initialize 4DVar 
    Var3d = True
    adjoint = False
    DA_4DVar = Var_Opt_PDE(forward_model=forward_model, 
                           observation_model=obs_transformation,
                           n_state=n_state, 
                           n_obs=n_obs, 
                           dt_obs=0.1, 
                           dt=0.01, 
                           Var3d=Var3d, 
                           adjoint=adjoint)

    num_mc = 5  
    # Load test data
    seq_state = np.load("../../data/KS_data_dim{}/test_seq_state.npy".format(n_state))[:num_mc, 500:]
    seq_obs = np.load("../../data/KS_data_dim{}/test_seq_obs.npy".format(n_state))[:num_mc, 500:]

    ass_w = 5
    T = 40*ass_w
    error_list = np.zeros(num_mc)
    time_list = np.zeros(num_mc)
    x_b = np.mean(seq_state, axis=(0,1))
    max_value = seq_state.max()
    min_value = seq_state.min()
    print("[INFO] Assimilation Window Length: {}".format(ass_w))
    print("[INFO] Traj Length: {}".format(T))
    for i in tqdm.tqdm(range(num_mc), desc="DA Evaulation"):
        traj_true = seq_state[i, :T]
        traj_estimation, eval_time = DA_4DVar.perform_4DVar(seq_obs[i], x_b=x_b, assimilation_window=ass_w, T=T)
        rmse = np.sqrt( np.mean( (np.array(traj_estimation) - traj_true)**2) )/(max_value-min_value)
        error_list[i] = rmse
        time_list[i] = eval_time
        print("[INFO] {}th traj Estimation Error: {}".format(i,rmse))


    fig, ax = plt.subplots(3, 1, figsize=(10, 24))
    ax[0].imshow(np.array(traj_estimation).T)
    ax[0].set_title('Estimated Trajectory')
    ax[1].imshow(traj_true.T)
    ax[1].set_title('True Trajectory')
    cbar = ax[2].imshow(np.abs(np.array(traj_estimation).T - traj_true.T))
    plt.colorbar(cbar, ax=ax[2])
    ax[2].set_title('Error')
    plt.show()
