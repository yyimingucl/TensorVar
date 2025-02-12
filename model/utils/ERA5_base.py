import torch 
import torch.nn as nn
import torch.nn.functional as F
import tltorch
import torch.utils
import torch.utils.data
from .utils import is_symmetric, weighted_MSELoss
import matplotlib.pyplot as plt
from torch import Tensor
            

# Forward Operator C_{S_{t+1}|S_t}
class forward_model(nn.Module):
    def __init__(self, phi_S:nn.Module, 
                       phi_inv_S:nn.Module, 
                       seq_length: int, 
                       *args, **kwargs) -> None:
        super(forward_model, self).__init__(*args, **kwargs)
        self.phi_S = phi_S
        self.phi_inv_S = phi_inv_S
        self.hidden_dim = phi_S.feature_dim
        self.seq_length = seq_length
        self.C_fwd = None
    
    def forward(self, state: torch.Tensor):
        z, encode_list = self.phi_S(state, return_encode_list=True)
        z_next = torch.mm(z, self.C_fwd)
        pred_s_next = self.phi_inv_S(z_next, encode_list)
        return pred_s_next

    def batch_latent_forward(self, batch_z: torch.Tensor):
        B = batch_z.shape[0]
        if self.C_fwd.dim() == 2:
            C_fwd = self.C_fwd.unsqueeze(0).repeat(B, 1, 1)
        else:
            C_fwd = self.C_fwd
        batch_z_next = torch.bmm(batch_z, C_fwd)
        return batch_z_next

    def latent_forward(self, z: torch.Tensor):
        z_next = torch.mm(z, self.C_fwd)
        return z_next

    
    def compute_loss(self, state_seq: torch.Tensor, state_next_seq: torch.Tensor, weight_matrix=None):
        B = state_seq.shape[0]
        device = state_seq.device
        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)
        
        encode_z_list = []

        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        loss_fwd = 0
        loss_identity = 0
        # loss_distance = torch.tensor(0.0).to(device)

        for i in range(self.seq_length):
            z_seq[:, i, :], encode_z_step_t = self.phi_S(state_seq[:, i, :], return_encode_list=True)
            encode_z_list.append(encode_z_step_t)

            z_next_seq[:, i, :] = self.phi_S(state_next_seq[:, i, :])


        z_seq_pinv = self.batch_pinv(z_seq)
        forward_weights = torch.bmm(z_seq_pinv, z_next_seq).mean(dim=0).repeat(B, 1, 1)

        self.C_fwd = forward_weights
        pred_z_next = self.batch_latent_forward(z_seq)
        
        
        for i in range(self.seq_length):
            recon_s = self.phi_inv_S(z_seq[:, i, :], encode_z_list[i])
            recon_s_next = self.phi_inv_S(pred_z_next[:, i, :], encode_z_list[i])

            # tmp_pred_s[:, i, :] = recon_s_next


            if weight_matrix is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, state_next_seq[:, i, :], weight_matrix).sum()
                loss_identity += weighted_MSELoss()(recon_s, state_seq[:, i, :], weight_matrix).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, state_next_seq[:, i, :])
                loss_identity += F.mse_loss(recon_s, state_seq[:, i, :])
                
        return loss_fwd, loss_identity, self.C_fwd.mean(dim=0)  
    
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
        z_seq_pinv = torch.linalg.solve(
            torch.bmm(z_seq_T, z_seq) + I_factor * I,
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
        
        BS = 256
        assert dynamics_dataset.seq_length == 1, "The sequence length of the dataset should be 1"
        
        dataloader = torch.utils.data.DataLoader(dynamics_dataset, batch_size=BS, shuffle=False)
        
        for i, batch_data in enumerate(dataloader):
            state, next_state = batch_data
            
            B = state.shape[0]
            
            state = state.to(device)
            next_state = next_state.to(device)
            state_feature = self.phi_S(state)
            next_state_feature = self.phi_S(next_state)
            forward_error = next_state_feature - self.batch_latent_forward(state_feature)
            for j in range(B):
                Q[i*B+j] = forward_error[j]
                B[i*B+j] = state_feature[j]
            
        Q = torch.cov(Q.T)
        B = torch.cov(B.T)
        
        # MPS does not support float64
        Q = Q.to("cpu")
        B = B.to("cpu")
        
        Q = torch.pinverse(Q + 0.1*torch.eye(self.hidden_dim))
        B = torch.pinverse(B + torch.eye(self.hidden_dim))
        
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
        plt.savefig(save_path + '/' + 'Q.png' )
        plt.close()
        
        plt.imshow(B.cpu().detach().numpy())
        plt.colorbar()
        plt.savefig(save_path + '/' + 'B.png' )
          
        if save_path is not None:
            save_path_Q = save_path + '/' + 'Q.pt'   
            print('[INFO] save Q to: ', save_path_Q)
            torch.save(Q, save_path_Q)
            
            save_path_B = save_path + '/' + 'B.pt'
            print('[INFO] save B to: ', save_path_B)
            torch.save(B, save_path_B)


class inverse_model(nn.Module):
    def __init__(self, phi_O:nn.Module, 
                       phi_S:nn.Module, 
                       phi_inv_S:nn.Module, 
                       *args, **kwargs) -> None:
        super(inverse_model, self).__init__(*args, **kwargs)
        self.phi_O = phi_O
        self.phi_S = phi_S
        self.phi_inv_S = phi_inv_S
        self.freeze_phi_S()
        self.freeze_phi_inv_S()
        
        self.C_S_O = None
     
    def forward(self, obs: Tensor):
         
        
        # state_analyzed = self.state_feature_decoder(state_feature_analyzed)
        # state_feature = self.state_feature_encoder(state_feature_analyzed)
        return self.phi_O(obs)
    
    def compute_loss(self, hist: Tensor, obs: Tensor, state: Tensor, weight_matrix=None):
        if weight_matrix is not None:
            pred_s = self.phi_O(obs)
            loss = weighted_MSELoss()(pred_s, state, weight_matrix).sum()
        else:
            pred_s = self.phi_O(obs)
            loss = F.mse_loss(pred_s, state)
        return loss
    
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
        print('[INFO] Saving DA_Model to: ', model_path)
        torch.save(self.state_dict(), model_path)


    def compute_R(self, da_dataset:torch.utils.data.Dataset, device:str="cpu", save_path:str=None):
         # R Compute the Covariance Matrix Cov(o_t,s_t)
         N = len(da_dataset)
         R = torch.zeros((N, self.phi_S.hidden_dims[-1])).to(device)
         
         dataloader = torch.utils.data.DataLoader(da_dataset, batch_size=256, shuffle=False)
         
         for i, batch_data in enumerate(dataloader):
             obs, state, hist = batch_data
             obs = obs.to(device)
             state = state.to(device)
             hist = hist.to(device)
             Bs = hist.shape[0]
             phi_S = self.phi_S(state)
             pred_phi_S = self.forward(obs=obs, hist=hist)
             obs_error = phi_S - pred_phi_S
             
             for j in range(Bs):
                 R[i*Bs+j] = obs_error[j]
                 
         R = torch.cov(R.T)    
         R = R.to("cpu")
         R = torch.pinverse(R + 1e-1*torch.eye(self.phi_S.hidden_dims[-1]))
         
         
         if not is_symmetric(R):
            print('[INFO] R is not symmetric, using symmetriziation')
            R = 0.5*(R + R.T)
         else:
            print('[INFO] R is symmetric')

         plt.imshow(R.cpu().detach().numpy())
         plt.colorbar()
         plt.savefig('R.png')
         plt.close()
          
          
         if save_path is not None: 
             save_path_R = save_path + '/' + 'R.pt' if not self.direct_sum else save_path + '/' + 'R_direct_sum.pt'  
             
             print('[INFO] save R to: ', save_path_R)    
             
             torch.save(R, save_path_R)
             