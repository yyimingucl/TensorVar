import torch 
import torch.nn as nn
import torch.nn.functional as F
import tltorch
import torch.utils
import torch.utils.data
from .utils.utils import is_symmetric
import matplotlib.pyplot as plt
from torch import Tensor
from einops import rearrange


# Featrues 
class phi_O_BASE(nn.Module):
    def __init__(self, featrues:nn.Module, *args, **kwargs) -> None:
        super(phi_O_BASE, self).__init__(*args, **kwargs)
        self.features = featrues
    
    def forward(self, x: torch.Tensor):
        return self.features(x)

class phi_H_BASE(nn.Module):
    def __init__(self, featrues:nn.Module, hist_w:int, *args, **kwargs) -> None:
        # hist_w is the history window length
        super(phi_H_BASE, self).__init__(*args, **kwargs)
        self.features = featrues
        self.hist_w = hist_w
    
    def forward(self, x: torch.Tensor):
        return self.features(x)

class phi_S_BASE(nn.Module):
    def __init__(self, featrues:nn.Module, *args, **kwargs) -> None:
        super(phi_S_BASE, self).__init__(*args, **kwargs)
        self.features = featrues
    
    def forward(self, x: torch.Tensor):
        return self.features(x)


class phi_inv_S_BASE(nn.Module):
    def __init__(self, featrues:nn.Module, *args, **kwargs) -> None:
        super(phi_inv_S_BASE, self).__init__(*args, **kwargs)
        self.features = featrues
    
    def forward(self, x: torch.Tensor):
        return self.features(x)


class forward_model(nn.Module):
    def __init__(self, phi_S:nn.Module, 
                       phi_inv_S:nn.Module, 
                       seq_length: int, 
                       hidden_dim: int=None,
                       *args, **kwargs) -> None:
        super(forward_model, self).__init__(*args, **kwargs)
        self.phi_S = phi_S
        self.phi_inv_S = phi_inv_S
        if hidden_dim is None:
            self.hidden_dim = phi_S.hidden_dims[-1]
        else:
            self.hidden_dim = hidden_dim
        assert seq_length > 5, "The sequence length should be greater than 5"
        self.train_length = int(2 * seq_length / 3)
        self.valid_length = seq_length - self.train_length
        self.seq_length = seq_length

        self.C_fwd = None
    
    def forward(self, state: torch.Tensor):
        z = self.phi_S(state)
        z_next = torch.mm(z, self.C_fwd)
        pred_s_next = self.phi_inv_S(z_next)
        return pred_s_next

    def batch_latent_forward(self, batch_z: torch.Tensor):
        B = batch_z.shape[0]
        if batch_z.dim() == 2:
            C_fwd = self.C_fwd
            batch_z_next = torch.mm(batch_z, C_fwd)
        else:
            if self.C_fwd.dim() == 2:
                C_fwd = self.C_fwd.unsqueeze(0).repeat(B, 1, 1)
            batch_z_next = torch.bmm(batch_z, C_fwd)
        return batch_z_next

    def latent_forward(self, z: torch.Tensor):
        z_next = torch.mm(z, self.C_fwd)
        return z_next
    
    def compute_loss(self, state_seq: torch.Tensor, state_next_seq: torch.Tensor):

        B = state_seq.shape[0]

        loss_fwd = 0
        loss_identity = 0
        if state_seq.ndim == 3:
            # [B, T, D]
            z_seq = self.phi_S(state_seq)
            z_next_seq = self.phi_S(state_next_seq)
        elif state_seq.ndim == 5:
            # [B, T, C, H, W]
            T = state_seq.shape[1]
            z_seq = []
            z_next_seq = []
            for i in range(T):
                z_seq.append(self.phi_S(state_seq[:,i,...]))
                z_next_seq.append(self.phi_S(state_next_seq[:,i,...]))
            z_seq = torch.stack(z_seq, dim=1)
            z_next_seq = torch.stack(z_next_seq, dim=1)
        z_seq_pinv = self.batch_pinv(z_seq)
        forward_weights = torch.bmm(z_seq_pinv, z_next_seq).mean(dim=0)

        self.C_fwd = forward_weights
        pred_z_next = self.batch_latent_forward(z_seq)
        

        recon_s = self.phi_inv_S(z_seq)
        recon_s_next = self.phi_inv_S(pred_z_next)
        if state_seq.ndim == 5:
            # [B, T, D]
            recon_s = rearrange(recon_s, '(B T) D W H-> B T D W H', B=B)
            recon_s_next = rearrange(recon_s_next, '(B T) D W H-> B T D W H', B=B)

        
        loss_fwd = F.mse_loss(recon_s_next, state_next_seq)
        loss_identity = F.mse_loss(recon_s, state_seq)
            
        return loss_fwd, loss_identity, self.C_fwd 
    
    @staticmethod
    def batch_pinv(z_seq: torch.Tensor, I_factor:float=1e-2):
        # inverse of z_seq
        # za_seq: [B, T, Dim_s]
        # I_factor: Identity factor
        B, T, D = z_seq.size()
        device = z_seq.device

        trans = T < D
        if trans:
            z_seq = torch.transpose(z_seq, 1, 2)
            T, D = D, T

        if not z_seq.is_cuda:
            z_seq = z_seq.to('cpu')
            I = torch.eye(D)[None, :, :].repeat(B, 1, 1).to('cpu')
        else:
            I = torch.eye(D)[None, :, :].repeat(B, 1, 1).to(device)
            
        z_seq_T = torch.transpose(z_seq, 1, 2)
        try:
            z_seq_pinv = torch.linalg.solve(
                torch.bmm(z_seq_T, z_seq) + I_factor * I,
                z_seq_T
            )
        except:
            z_seq_pinv = torch.linalg.solve(
                torch.bmm(z_seq_T, z_seq) + 10 * I,
                z_seq_T
            )
        if trans:
            z_seq_pinv = torch.transpose(z_seq_pinv, 1, 2)

        return z_seq_pinv.to(device)
    
    def save_C_fwd(self, path, C_fwd):
        C_fwd_filename = path + '/' + 'C_fwd.pt'
        print('[INFO] Saving C_fwd weights to:', C_fwd_filename)
        torch.save(C_fwd, C_fwd_filename)
        
    def save_model(self, path):
        # Save the model
        self.to('cpu')
        filename = path + '/forward_model.pt'
        print('[INFO] Saving forward_model weights to:', filename)
        torch.save(self.state_dict(), filename)

    def compute_Q_B(self, dynamics_dataset:torch.utils.data.Dataset, device:str='cpu', save_path:str=None):
        # Compute the Covariance Matrix Cov(s_t,s_{t+1})
        # Compute the Covariance Matrix Cov(s_t,s_t)
        N = len(dynamics_dataset)
        Q = torch.zeros((N, self.hidden_dim)).to(device)
        B = torch.zeros((N, self.hidden_dim)).to(device)
        
        BS = 2048
        assert dynamics_dataset.seq_length == 1, "The sequence length of the dataset should be 1"
        
        dataloader = torch.utils.data.DataLoader(dynamics_dataset, batch_size=BS, shuffle=False)
        
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                state, next_state = batch_data
                
                bs = state.shape[0]
                
                state = state.to(device)
                next_state = next_state.to(device)
                state_feature = self.phi_S(state)
                next_state_feature = self.phi_S(next_state)
                forward_error = next_state_feature - self.batch_latent_forward(state_feature)
                for j in range(bs):
                    Q[i*bs+j] = forward_error[j]
                    B[i*bs+j] = state_feature[j]
            
        Q = torch.cov(Q.T)
        B = torch.cov(B.T)
        
        # MPS does not support float64
        Q = Q.to("cpu")
        B = B.to("cpu")
        
        Q = torch.pinverse(Q + 0.1*torch.eye(self.hidden_dim))
        B = torch.pinverse(B + 0.1*torch.eye(self.hidden_dim))
        
        if not is_symmetric(Q):
            print('[INFO] Q is not symmetric, using symmetriziation')
            Q = 0.5*(Q + Q.T)
        else:
            print('[INFO] Q is symmetric')
            
        if not is_symmetric(B):
            print('[INFO] B is not symmetric, using symmetriziation')
            B = 0.5*(B + B.T)
        else:
            print('[INFO] B is symmetric')
            
        plt.imshow(Q.cpu().detach().numpy())
        plt.colorbar()
        plt.savefig(save_path + '/Q.png')
        plt.close()
        
        plt.imshow(B.cpu().detach().numpy())
        plt.colorbar()
        plt.savefig(save_path + '/B.png')
        plt.close()
          
        if save_path is not None:
            save_path_Q = save_path + '/' + 'Q.pt'   
            print('[INFO] save Q to: ', save_path_Q)
            torch.save(Q, save_path_Q)
            
            save_path_B = save_path + '/' + 'B.pt'
            print('[INFO] save B to: ', save_path_B)
            torch.save(B, save_path_B)
            
            


class inverse_model(nn.Module):
     def __init__(self, 
                  phi_O:nn.Module, 
                  phi_H:nn.Module, 
                  phi_S:nn.Module, 
                  phi_inv_S:nn.Module, 
                  rate:tuple[float, float, float],
                  *args, **kwargs) -> None:
        super(inverse_model, self).__init__(*args, **kwargs)
        self.phi_O = phi_O
        self.phi_H = phi_H
        self.phi_S = phi_S
        self.phi_inv_S = phi_inv_S
        self.freeze_phi_S()
        self.freeze_phi_inv_S()

        assert rate[0] > 0 and rate[1] > 0 and rate[2] > 0, "The rate should be greater than 0"
        assert rate[0] <= 1 and rate[1] <= 1 and rate[2] <= 1, "The rate should be less than or equal to 1"
        rank = (int(phi_H.hidden_dims[-1] * rate[2]), int(phi_O.hidden_dims[-1] * rate[1]), int(phi_S.hidden_dims[-1] * rate[0]))
        self.C_invobs = tltorch.TRL(input_shape=(phi_H.hidden_dims[-1], phi_O.hidden_dims[-1]), 
                                    output_shape=(phi_S.hidden_dims[-1]),
                                    factorization='tucker', rank=rank)

     
     def forward(self, obs: Tensor, hist: Tensor):
         
        phi_o = self.phi_O(obs)
        phi_h = self.phi_H(hist)
        phi_h = phi_h.unsqueeze(2)
        phi_o = phi_o.unsqueeze(1)
        phi_O_tensor_K_h = phi_h * phi_o
            
        phi_s = self.C_invobs(phi_O_tensor_K_h)
        return phi_s
    
     def compute_loss(self, hist: Tensor, obs: Tensor, state: Tensor):
         pred_phi_s = self.forward(obs, hist)
         phi_s = self.phi_S(state)
         loss_1 = F.mse_loss(pred_phi_s, phi_s)
         loss_2 = F.mse_loss(self.phi_inv_S(phi_s), state)
         return loss_1, loss_2
    
     def freeze_phi_S(self):
         print('[INFO] Freezing phi_S')
         for param in self.phi_S.parameters():
             param.requires_grad = False
    
     def defreeze_phi_S(self):
         print('[INFO] Defreezing phi_S')
         for param in self.phi_S.parameters():
             param.requires_grad = True
             
     def freeze_phi_inv_S(self):
            print('[INFO] Freezing phi_inv_S')
            for param in self.phi_inv_S.parameters():
                param.requires_grad = False
    
     def defreeze_phi_inv_S(self):
            print('[INFO] Defreezing phi_inv_S')
            for param in self.phi_inv_S.parameters():
                param.requires_grad = True
    
     def save_model(self, path):
        self.to('cpu')
        model_path = path + '/' + 'inv_obs_model.pt'
        print('[INFO] Saving inv_obs_model to: ', model_path)
        torch.save(self.state_dict(), model_path)


     def compute_R(self, 
                   inv_obs_dataset:torch.utils.data.Dataset, 
                   device:str="cpu", 
                   save_path:str=None):
         # R Compute the Covariance Matrix Cov(o_t,s_t)
         N = len(inv_obs_dataset)
         R = torch.zeros((N, 
                          self.phi_S.hidden_dims[-1])).to(device)
         
         dataloader = torch.utils.data.DataLoader(inv_obs_dataset, 
                                                  batch_size=32, 
                                                  shuffle=False)
         with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                obs, state, hist = batch_data
                obs = obs.to(device)
                state = state.to(device)
                hist = hist.to(device)
                Bs = hist.shape[0]
                phi_s = self.phi_S(state)
                pred_phi_s = self.forward(obs=obs, hist=hist)
                obs_error = phi_s - pred_phi_s
                
                for j in range(Bs):
                    R[i*Bs+j] = obs_error[j]
                 
         R = torch.cov(R.T)    
         R = R.to("cpu")
         R = torch.pinverse(R + 0.01*torch.eye(self.phi_S.hidden_dims[-1]))
         
         
         if not is_symmetric(R):
            print('[INFO] R is not symmetric, using symmetriziation')
            R = 0.5*(R + R.T)
         else:
            print('[INFO] R is symmetric')

         plt.imshow(R.cpu().detach().numpy())
         plt.colorbar()
         plt.savefig(save_path + '/R.png')
         plt.close()
          
         if save_path is not None:
             save_path_R = save_path + '/' + 'R.pt' 
             print('[INFO] save R to: ', save_path_R)    
             torch.save(R, save_path_R)
             


class inverse_model_2D(nn.Module):
     def __init__(self, 
                  phi_OH:nn.Module, 
                  phi_S:nn.Module, 
                  phi_inv_S:nn.Module, 
                  *args, **kwargs) -> None:
        super(inverse_model, self).__init__(*args, **kwargs)
        self.phi_OH = phi_OH
        self.phi_S = phi_S
        self.phi_inv_S = phi_inv_S
        self.freeze_phi_S()
        self.freeze_phi_inv_S()
        self.C_invobs = nn.Linear(phi_OH.feature_dim, phi_S.feature_dim, bias=False)

     
     def forward(self, obs_hist: Tensor):
        phi_O_tensor_K_h = self.phi_OH(obs_hist)   
        phi_s = self.C_invobs(phi_O_tensor_K_h)
        return phi_s
    
     def compute_loss(self, obs_hist: Tensor, state: Tensor):
         pred_phi_s = self.forward(obs_hist)
         phi_s = self.phi_S(state)
         loss_1 = F.mse_loss(pred_phi_s, phi_s)
         loss_2 = F.mse_loss(self.phi_inv_S(phi_s), state)
         return loss_1, loss_2
    
     def freeze_phi_S(self):
         print('[INFO] Freezing phi_S')
         for param in self.phi_S.parameters():
             param.requires_grad = False
    
     def defreeze_phi_S(self):
         print('[INFO] Defreezing phi_S')
         for param in self.phi_S.parameters():
             param.requires_grad = True
             
     def freeze_phi_inv_S(self):
            print('[INFO] Freezing phi_inv_S')
            for param in self.phi_inv_S.parameters():
                param.requires_grad = False
    
     def defreeze_phi_inv_S(self):
            print('[INFO] Defreezing phi_inv_S')
            for param in self.phi_inv_S.parameters():
                param.requires_grad = True
    
     def save_model(self, path):
        self.to('cpu')
        model_path = path + '/' + 'inv_obs_model.pt'
        print('[INFO] Saving inv_obs_model to: ', model_path)
        torch.save(self.state_dict(), model_path)


     def compute_R(self, 
                   inv_obs_dataset:torch.utils.data.Dataset, 
                   device:str="cpu", 
                   save_path:str=None):
         # R Compute the Covariance Matrix Cov(o_t,s_t)
         N = len(inv_obs_dataset)
         R = torch.zeros((N, 
                          self.phi_S.feature_dim)).to(device)
         
         dataloader = torch.utils.data.DataLoader(inv_obs_dataset, 
                                                  batch_size=256, 
                                                  shuffle=False)
         with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                obs_hist, state, _ = batch_data
                obs = obs.to(device)
                state = state.to(device)
                Bs = state.shape[0]
                phi_s = self.phi_S(state)
                pred_phi_s = self.forward(obs_hist)
                obs_error = phi_s - pred_phi_s
                
                for j in range(Bs):
                    R[i*Bs+j] = obs_error[j]
                 
         R = torch.cov(R.T)    
         R = R.to("cpu")
         R = torch.pinverse(R + 0.01*torch.eye(self.phi_S.feature_dim))
         
         
         if not is_symmetric(R):
            print('[INFO] R is not symmetric, using symmetriziation')
            R = 0.5*(R + R.T)
         else:
            print('[INFO] R is symmetric')

         plt.imshow(R.cpu().detach().numpy())
         plt.colorbar()
         plt.savefig(save_path + '/R.png')
         plt.close()
          
         if save_path is not None:
             save_path_R = save_path + '/' + 'R.pt' 
             print('[INFO] save R to: ', save_path_R)    
             torch.save(R, save_path_R)
             