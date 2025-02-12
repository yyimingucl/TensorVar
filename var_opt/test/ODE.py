import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import tqdm
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit
from jax import random as jax_random

from LBFGS_Var import Var_Opt


if __name__ == "__main__":

    # Example: Lorenz96
    # forward model
    @jit
    def lorenz96(t, X, args):
        F, = args
        X_right_shift = jnp.roll(X, 1)
        X_right_shift2 = jnp.roll(X, 2)
        X_left_shift = jnp.roll(X, -1)
        dX = -X_right_shift * (X_right_shift2 - X_left_shift) - X + F
        return dX

    n_state = 40 # 40

    obs_index = np.linspace(0, n_state, int(n_state * 0.2), dtype=int)
    obs_index[-1] = n_state - 1
    obs_index = jnp.array(obs_index)

    n_obs = len(obs_index)

    # observation model
    @jit
    def obs_transformation(x):
        key = jax_random.PRNGKey(2024)
        return 5 * jnp.arctan(x[obs_index] * 0.1) + 0.1 * jax_random.normal(key, shape=x[obs_index].shape)

    # 4D-Var
    DA_4DVar = Var_Opt(forward_ODE=lorenz96, observation_model=obs_transformation, ODE_arg=(10,),
                        n_state=n_state, n_obs=n_obs, dt_obs=0.1, dt=0.01, Var3d=False, adjoint=False)

    num_mc = 5  
    # Load test data
    seq_state = np.load("../../data/L96_data_dim{}/test_seq_state.npy".format(n_state))[:num_mc, 500:]
    seq_obs = np.load("../../data/L96_data_dim{}/test_seq_obs.npy".format(n_state))[:num_mc, 500:]

    ass_w = 3
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