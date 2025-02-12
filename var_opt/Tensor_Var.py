import numpy as np
import tqdm 
import time
import cvxpy as cp
import torch
from cvxpy import quad_form
import numpy as np

def QP_solver(z_b, seq_z, F, B, R, Q, T):
    """
    Quadratic Programming Solver
    """
    constraints = []
    state_dim = z_b.shape[0]
    pred_seq_z = cp.Variable((T, state_dim))
    cost = 0 
    
    penalty = np.linspace(0.1, 1, T-1)[::-1]
    for t in range(T):
        if t == 0:
            cost += quad_form(pred_seq_z[t] - z_b, B)
            cost += quad_form(pred_seq_z[t] - seq_z[t], R)
            
        else:   
            cost += quad_form(pred_seq_z[t] - seq_z[t], R)
            cost += quad_form(pred_seq_z[t] - F.T@pred_seq_z[t-1], penalty[t-1]*Q)
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective, constraints)
    prob.solve() 
    # Get the final solution
    analyzed_traj_states = pred_seq_z.value
    
    return analyzed_traj_states

class Tensor_Var:
    def __init__(self, forward_model, observation_model, n_state, n_obs, B, R, Q) -> None:
        self.forward_model = forward_model
        self.observation_model = observation_model
        self.n_state = n_state
        self.n_obs = n_obs
        try:
            if hasattr(self.forward_model, "phi_S"):
                self.n_hidden = self.forward_model.phi_S.hidden_dims[-1]
            else:
                self.n_hidden = self.forward_model.K_S.hidden_dims[-1]
        except:
            if hasattr(self.forward_model, "phi_S"):
                self.n_hidden = self.forward_model.phi_S.feature_dim
            else:
                self.n_hidden = self.forward_model.K_S.feature_dim

        self.B = B
        self.R = R
        self.Q = Q
        
    def perform_4DVar(self, 
                           seq_obs:np.array, 
                           seq_hist:np.array,
                           z_b, 
                           assimilation_window:int=3, 
                           T:int=500):
        
        B = self.B
        R = self.R
        Q = self.Q

        F = self.forward_model.C_fwd.detach().numpy()
        
        z_b = z_b.cpu().numpy()

        horizon_traj_estimation = []
        start_time = time.time()
        with torch.no_grad():
            for start in tqdm.tqdm(range(0, T, assimilation_window)):
                end = min(start + assimilation_window, T)
                # project the phi_o, phi_h to phi_s
                hist = torch.Tensor(seq_hist[start:end]).permute(0, 2, 1)
                obs = torch.Tensor(seq_obs[start:end])
                seq_z = self.observation_model.forward(hist=hist, obs=obs).numpy()
                result = QP_solver(z_b=z_b, 
                                  seq_z=seq_z, 
                                  F=F, 
                                  B=B, 
                                  Q=Q, 
                                  R=R, 
                                  T=assimilation_window)

                optimal_traj = self.forward_model.phi_inv_S(torch.Tensor(result)).numpy()
                horizon_traj_estimation.extend(optimal_traj)
        end_time = time.time()
        evaluation_time = end_time - start_time
        return horizon_traj_estimation, evaluation_time
    
    def perform_4DVar_ERA5(self, 
                           seq_obs_hist:np.array, 
                           z_b, 
                           assimilation_window:int=3, 
                           T:int=500,
                           device="cuda"):    
        
        B = self.B
        R = self.R
        Q = self.Q
        try:
            F = self.forward_model.C_fwd.detach().cpu().numpy()
        except:
            F = self.forward_model.C_forward.detach().cpu().numpy()

        z_b = z_b.cpu().numpy()

        horizon_traj_estimation = []
        start_time = time.time()  
        with torch.no_grad():
            for start in tqdm.tqdm(range(0, T, assimilation_window)):
                end = min(start + assimilation_window, T)
                # project the phi_o, phi_h to K_S
                obs_hist = torch.Tensor(seq_obs_hist[start:end])
                seq_z = self.observation_model.forward(obs_hist)    #.cpu().numpy()
                seq_z, en_list = self.forward_model.K_S(seq_z, 
                                                        return_encode_list=True)
                seq_z = seq_z.detach().cpu().numpy()
                result = QP_solver(z_b=z_b, 
                                   seq_z=seq_z, 
                                   F=F, 
                                   B=B, 
                                   Q=Q, 
                                   R=R, 
                                   T=assimilation_window)
                optimal_traj = self.forward_model.K_S_preimage(torch.Tensor(result).to(seq_obs_hist.device), en_list).detach().cpu().numpy()
                horizon_traj_estimation.extend(optimal_traj)
        end_time = time.time()
        evaluation_time = end_time - start_time
        return horizon_traj_estimation, evaluation_time