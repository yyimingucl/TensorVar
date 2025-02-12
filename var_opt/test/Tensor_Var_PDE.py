import os 
import sys
home_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(home_dir)

import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.KS_model_D256 import KS_C_FORWARD
from var_opt.Tensor_Var import Tensor_Var


if __name__ == "__main__":

    # Set the parameters
    n_state = 256
    n_obs = int(n_state * 0.2)

    ass_w = 5 # assimilation window 
    T = 40*ass_w

    model_save_path = '../../model_weights/KS_state_dim{}_transformer'.format(n_state)

    # Load models
    forward_model = KS_C_FORWARD()
    forward_model.load_state_dict(torch.load(model_save_path + '/forward_model.pt', map_location="cpu"))
    forward_model.C_forward = torch.load(model_save_path + '/C_forward.pt', map_location="cpu")

    # inv_obs_model = KS_C_INVOBS()
    # inv_obs_model.load_state_dict(torch.load(model_save_path + '/inv_obs_model.pt', map_location="cpu"))

    inv_obs_model = torch.load(model_save_path + '/inv_obs_model.pt', map_location="cpu")
    # inv_obs_model.to("cpu")
    # Load error covariance matrices
    # Q = torch.load(model_save_path + '/' + 'Q.pt', map_location="cpu")
    # R = torch.load(model_save_path + '/' + 'R.pt', map_location="cpu")
    # B = torch.load(model_save_path + '/' + 'B.pt', map_location="cpu")

    hidden_dim = forward_model.K_S.hidden_dims[-1] 
    B = np.eye(hidden_dim) * 0.01
    R = np.eye(hidden_dim) 
    Q = np.eye(hidden_dim) * 0.1

    DA_Tensor_Var = Tensor_Var(forward_model=forward_model, 
                               observation_model=inv_obs_model, 
                               n_state=n_state, 
                               n_obs=n_obs,
                               B=B,
                               Q=Q,
                               R=R)


    num_mc = 5  
    # Load test data
    seq_state = np.load("../../data/KS_data_dim{}/test_seq_state.npy".format(n_state))[:num_mc, 500:]
    seq_obs = np.load("../../data/KS_data_dim{}/test_seq_obs.npy".format(n_state))[:num_mc, 500:]
    seq_hist = np.load("../../data/KS_data_dim{}/test_seq_hist.npy".format(n_state))[:num_mc, 500:]

   
    error_list = np.zeros(num_mc)
    time_list = np.zeros(num_mc)
    x_b = np.mean(seq_state, axis=(0,1))
    max_value = seq_state.max()
    min_value = seq_state.min()
    print("[INFO] Assimilation Window Length: {}".format(ass_w))
    print("[INFO] Traj Length: {}".format(T))
    for i in tqdm.tqdm(range(num_mc), desc="DA Evaulation"):
        traj_true = seq_state[i, :T]
        traj_estimation, eval_time = DA_Tensor_Var.perform_4DVar(seq_obs[i], 
                                                                 seq_hist=seq_hist[i], 
                                                                 x_b=x_b, 
                                                                 assimilation_window=ass_w, 
                                                                 T=T)
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

    


