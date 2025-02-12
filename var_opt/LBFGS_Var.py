'''
file:          LBFGS_VAR.py
author:        yyimingucl <yyiming3@gmail.com>
date:          2025-01-23 12:09:03
description:   Model-based 3D/4D-Var 
               weak-constraint 4D-Var optimized by L-BFGS-B implementation using JAX .
'''
import numpy as np
import time
import tqdm

import jax.numpy as jnp
from jax import jit, value_and_grad
from jax import random as jax_random
from jaxopt import ScipyMinimize
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

"""
4D-Var optimization using L-BFGS-B for ODE systems
"""

class Var_Opt:
    def __init__(self, 
                 forward_ODE, 
                 observation_model, 
                 ODE_arg:tuple, 
                 n_state:int, 
                 n_obs:int, 
                 dt_obs:float=0.1, 
                 dt:float=0.01, 
                 Var3d=False, 
                 adjoint:bool=False) -> None:
        
        self.forward_ODE = forward_ODE
        self.forward_model_args = ODE_arg
        self.observation_model = observation_model
        
        # Default setting for 4D-Var
        self.R_inv = jnp.eye(n_obs)
        self.B_inv = jnp.eye(n_state) * 0.1
        self.Q_inv = jnp.eye(n_state) * 0.1
        
        self.n_state = n_state
        self.n_obs = n_obs
        self.dt_obs = dt_obs
        self.dt = dt
        
        self.solver = Tsit5()
        # forward model
        self.forward_model = self.create_forward_model()
        
        self.Var3d = Var3d
        
        # cost function
        cost_function = self.generate_cost_function()
        
        # adjoint gradient
        if adjoint:
            assert Var3d == False, "Adjoint gradient is not implemented for 3D-Var"
            self.adjoint = True
            # adjoint gradient 
            # the jax.value_and_grad compute thevalue and gradient in respect to the first argument 'argnums=0'
            # it computes the gradient in a reverse mode by default and this is equivalent to the adjoint method
            # Ref https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L403-L477
            self.cost_function = cost_function
            self.cost_function_and_grad = value_and_grad(cost_function, argnums=0)
        else:
            self.adjoint = False
            self.cost_function = cost_function
            self.cost_function_and_grad = False

    def create_forward_model(self):     
        forward_ODE = self.forward_ODE
        solver = self.solver 
        args = self.forward_model_args
        
        @jit
        def forward_model(x:jnp.array):
            term = ODETerm(forward_ODE)
            t0 = 0
            t1 = self.dt_obs
            saveat = SaveAt(ts=jnp.array([t1]))  # Time points for observation
            sol = diffeqsolve(term, solver, t0, t1, self.dt, x, args, saveat=saveat, max_steps=int(1e3))
            return sol.ys.squeeze()
    
        return forward_model


    def generate_cost_function(self):
        forward_model = self.forward_model
        observation_model = self.observation_model
        Var3d = self.Var3d
        if Var3d:
            @jit
            def cost_function(trajectory:jnp.array, x_b:jnp.array, y_o:jnp.array, 
                            R_inv:jnp.array, B_inv:jnp.array, Q_inv:jnp.array):
                n_steps = len(y_o)
                trajectory = trajectory.reshape((n_steps, -1))
                x0 = trajectory[0]
                cost = 0.5 * jnp.dot((x0 - x_b).T, jnp.dot(B_inv, (x0 - x_b)))
                for i in range(n_steps):
                    y_model = observation_model(trajectory[i])                   
                    cost += 0.5 * jnp.dot((y_o[i] - y_model).T, jnp.dot(R_inv, (y_o[i] - y_model)))
                return cost
        else:
            def cost_function(trajectory:jnp.array, x_b:jnp.array, y_o:jnp.array, 
                            R_inv:jnp.array, B_inv:jnp.array, Q_inv:jnp.array):
                n_steps = len(y_o)
                trajectory = trajectory.reshape((n_steps, -1))
                x0 = trajectory[0]
                cost = 0.5 * jnp.dot((x0 - x_b).T, jnp.dot(B_inv, (x0 - x_b)))
                for i in range(n_steps):
                    x = trajectory[i]
                    if i <= n_steps - 1:
                        x_next = trajectory[i + 1]
                        model_error = x_next - forward_model(x)        
                        cost += 0.5 * jnp.dot(model_error.T, jnp.dot(Q_inv, model_error))
                    y_model = observation_model(x)
                    cost += 0.5 * jnp.dot((y_o[i] - y_model).T, jnp.dot(R_inv, (y_o[i] - y_model)))
                return cost
        return cost_function

    def background_traj(self, x_b:jnp.array, n_steps:int):
        args = self.forward_model_args
        term = ODETerm(self.forward_ODE)
        t0 = 0
        t1 = n_steps * self.dt_obs
        saveat = SaveAt(ts=jnp.arange(t0, t1, self.dt_obs))  # Time points for observation
        sol = diffeqsolve(term, self.solver, t0, t1, self.dt, x_b, args, saveat=saveat, max_steps=int(1e9))
        if n_steps <= 3:
            return sol.ys[:-1]
        else:
            return sol.ys

    def optimize(self, x_b:jnp.array, y_o:jnp.array, n_steps:int):       
        # Initial guess
        traj = self.background_traj(x_b, n_steps).flatten()
        result = ScipyMinimize(method='L-BFGS-B', fun=self.cost_function, maxiter=100000, 
                               options={'ftol': 1e-12}, 
                               value_and_grad=self.cost_function_and_grad).run(traj, x_b, y_o, self.R_inv, self.B_inv, self.Q_inv)
        print("[INFO] Number of optimizer iteration: ", result.state.iter_num)
        return result.params.reshape((n_steps, self.n_state))

    def perform_4DVar(self, seq_obs:jnp.array, x_b:jnp.array=None, assimilation_window:int=3, T:int=500) -> tuple:
        # seq_obs: the sequence of observations
        # x_b: the background state
        # Assimilation window: the horizon of the 4D-Var
        # T: the length of the trajectory to be estimated

        key = jax_random.PRNGKey(2024)
        if x_b is None:
            x_b = jax_random.normal(key, shape=(self.n_state,))
        else:
            assert x_b.shape == (self.n_state,)
        horizon_traj_estimation = []
        start_time = time.time()
        for start in range(0, T, assimilation_window):
            end = min(start + assimilation_window, T)
            y_o_window = jnp.array(seq_obs[start:end])
            optimal_traj = self.optimize(x_b, y_o_window, end - start)
            horizon_traj_estimation.extend(optimal_traj)
            
        end_time = time.time()
        evaluate_time = end_time - start_time
        return horizon_traj_estimation, evaluate_time


"""
4D-Var optimization using L-BFGS-B for PDE systems
"""
class Var_Opt_PDE:
    def __init__(self, forward_model, 
                       observation_model, 
                       n_state:int, 
                       n_obs:int, 
                       dt_obs:float=0.1, 
                       dt:float=0.01, 
                       Var3d=False,
                       adjoint:bool=False) -> None:
        self.forward_model = forward_model
        self.observation_model = observation_model
        
        # Default setting for 4D-Var
        self.R_inv = jnp.eye(n_obs)
        self.B_inv = jnp.eye(n_state) * 0.01
        self.Q_inv = jnp.eye(n_state) * 0.1
        
        self.n_state = n_state
        self.n_obs = n_obs
        self.dt_obs = dt_obs
        self.dt = dt

        self.Var3d = Var3d
        
        # cost function
        cost_function = self.generate_cost_function()
        
        # adjoint gradient
        if adjoint:
            assert Var3d == False, "Adjoint gradient is not implemented for 3D-Var"
            self.adjoint = True
            # adjoint gradient 
            # the jax.value_and_grad compute thevalue and gradient in respect to the first argument 'argnums=0'
            # it computes the gradient in a reverse mode by default and this is equivalent to the adjoint method
            # Ref https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L403-L477
            self.cost_function = cost_function
            self.cost_function_and_grad = value_and_grad(cost_function, argnums=0)
        else:
            self.adjoint = False
            self.cost_function = cost_function
            self.cost_function_and_grad = False
    
    def generate_cost_function(self):
        forward_model = self.forward_model
        observation_model = self.observation_model
        
        Var3d = self.Var3d
        
        if Var3d:
            @jit
            def cost_function(trajectory:jnp.array, x_b:jnp.array, y_o:jnp.array, 
                            R_inv:jnp.array, B_inv:jnp.array, Q_inv:jnp.array):
                n_steps = len(y_o)
                trajectory = trajectory.reshape((n_steps + 1, -1))
                x0 = trajectory[0]
                cost = 0.5 * jnp.dot((x0 - x_b).T, jnp.dot(B_inv, (x0 - x_b)))
                for i in range(n_steps):
                    x_next = trajectory[i + 1]
                    y_model = observation_model(x_next)                   
                    cost += 0.5 * jnp.dot((y_o[i] - y_model).T, jnp.dot(R_inv, (y_o[i] - y_model)))
                return cost
        else:
            @jit
            def cost_function(trajectory:jnp.array, x_b:jnp.array, y_o:jnp.array, 
                            R_inv:jnp.array, B_inv:jnp.array, Q_inv:jnp.array):
                n_steps = len(y_o)
                trajectory = trajectory.reshape((n_steps + 1, -1))
                x0 = trajectory[0]
                cost = 0.5 * jnp.dot((x0 - x_b).T, jnp.dot(B_inv, (x0 - x_b)))
                for i in range(n_steps):
                    x = trajectory[i]
                    x_next = trajectory[i + 1]
                    model_error = x_next - forward_model(x, 1)[-1]
        
                    y_model = observation_model(x_next)
                    
                    cost += 0.5 * jnp.dot((y_o[i] - y_model).T, jnp.dot(R_inv, (y_o[i] - y_model)))
                    cost += 0.5 * jnp.dot(model_error.T, jnp.dot(Q_inv, model_error))
                return cost
        
        return cost_function

    def background_traj(self, x_b:jnp.array, n_steps:int):
        return self.forward_model(x_b, n_steps)

    def optimize(self, x_b:jnp.array, y_o:jnp.array, n_steps:int):       
        # Initial guess
        traj = self.background_traj(x_b, n_steps).flatten()
        result = ScipyMinimize(method='L-BFGS-B', 
                               fun=self.cost_function, 
                               maxiter=100000,  
                               value_and_grad=self.cost_function_and_grad).run(traj, 
                                                                               x_b,
                                                                               y_o, 
                                                                               self.R_inv, 
                                                                               self.B_inv, 
                                                                               self.Q_inv)

        return result.params.reshape((n_steps+1, self.n_state))

    def perform_4DVar(self, 
                      seq_obs:jnp.array, 
                      x_b:jnp.array=None, 
                      assimilation_window:int=3, 
                      T:int=500) -> tuple:
        # seq_obs: the sequence of observations
        # x_b: the background state
        # Assimilation window: the horizon of the 4D-Var
        # T: the length of the trajectory to be estimated

        key = jax_random.PRNGKey(2024)
        if x_b is None:
            x_b = jax_random.normal(key, shape=(self.n_state,))
        else:
            assert x_b.shape == (self.n_state,)
        horizon_traj_estimation = []
        start_time = time.time()
        for start in range(0, T, assimilation_window):
            end = min(start + assimilation_window, T)
            y_o_window = jnp.array(seq_obs[start:end])
            optimal_traj = self.optimize(x_b, y_o_window, end - start)
            horizon_traj_estimation.extend(optimal_traj[1:])
            x_b = optimal_traj[-1]
            
        end_time = time.time()
        evaluate_time = end_time - start_time
        return horizon_traj_estimation, evaluate_time


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num_mc = 5  
    ass_w = 3
    T = 40*ass_w

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
    
    n_state = 80 # 40
    
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
                       n_state=n_state, n_obs=n_obs, dt_obs=0.1, dt=0.01, Var3d=True)


    # Load test data
    seq_state = np.load("../data/L96_data_dim{}/test_seq_state.npy".format(n_state))[:num_mc, 500:]
    seq_obs = np.load("../data/L96_data_dim{}/test_seq_obs.npy".format(n_state))[:num_mc, 500:]

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