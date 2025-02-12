'''
file:          base_model.py
author:        yyimingucl <yyiming3@gmail>
date:          2024-06-21 18:38:41
description:   Base model for RKHS_DA
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
import tltorch
import torch.utils
import torch.utils.data
from .utils import is_symmetric, weighted_MSELoss
import matplotlib.pyplot as plt
from torch import Tensor

# Featrues 
class K_O_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, *args, **kwargs) -> None:
        super(K_O_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)

class K_H_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, hist_w:int, *args, **kwargs) -> None:
        # hist_w is the history window length
        super(K_H_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
        self.hist_w = hist_w
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)

class K_S_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, *args, **kwargs) -> None:
        super(K_S_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)


class K_S_preimage_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, *args, **kwargs) -> None:
        super(K_S_preimage_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)


# Forward Operator C_{S_{t+1}|S_t}
class forward_model(nn.Module):
    def __init__(self, K_S:nn.Module, K_S_preimage:nn.Module, 
                       seq_length: int, *args, **kwargs) -> None:
        super(forward_model, self).__init__(*args, **kwargs)
        self.K_S = K_S
        self.K_S_preimage = K_S_preimage
        self.hidden_dim = K_S.hidden_dims[-1]
        self.seq_length = seq_length
        self.C_forward = None
    
    def forward(self, state: torch.Tensor):
        z = self.K_S(state)
        z_next = torch.mm(z, self.C_forward)
        pred_s_next = self.K_S_preimage(z_next)
        return pred_s_next

    def batch_latent_forward(self, batch_z: torch.Tensor):
        # print('batch_z:', batch_z.shape)
        # print('C_forward:', self.C_forward.shape)
        # batch_z_next = torch.bmm(self.C_forward, batch_z.permute(0,2,1)).squeeze(-1).permute(0,2,1)
        B = batch_z.shape[0]
        if self.C_forward.dim() == 2:
            C_forward = self.C_forward.unsqueeze(0).repeat(B, 1, 1)
        else:
            C_forward = self.C_forward
        batch_z_next = torch.bmm(batch_z, C_forward)
        return batch_z_next

    def latent_forward(self, z: torch.Tensor):
        z_next = torch.mm(z, self.C_forward)
        return z_next
    
    def compute_loss(self, state_seq: torch.Tensor, state_next_seq: torch.Tensor):
        B = state_seq.shape[0]
        device = state_seq.device
        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)
        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        loss_fwd = 0
        loss_identity = 0
        loss_distance = torch.tensor(0.0).to(device)

        for i in range(self.seq_length):
            z_seq[:, i, :] = self.K_S(state_seq[:, i, :])
            z_next_seq[:, i, :] = self.K_S(state_next_seq[:, i, :])

        z_seq_pinv = self.batch_pinv(z_seq[:, :-2, :])
        forward_weights = torch.bmm(z_seq_pinv, z_next_seq[:, :-2, :]).mean(dim=0).repeat(B, 1, 1)

        self.C_forward = forward_weights
        pred_z_next = self.batch_latent_forward(z_seq[:, -2:, :])
        
        # tmp_pred_s = torch.zeros(B, self.seq_length, 40).to(device)
        
        for i in range(self.seq_length):
            recon_s = self.K_S_preimage(z_seq[:, i, :])
            loss_identity += F.mse_loss(recon_s, state_seq[:, i, :])

            if i >= self.seq_length - 2:
                    recon_s_next = self.K_S_preimage(pred_z_next[:, i-(self.seq_length - 2), :])
                    loss_fwd += F.mse_loss(recon_s_next, state_next_seq[:, i, :])
            else:
                continue
        # with torch.no_grad():
        #     tmp_pred_s = tmp_pred_s.cpu().detach()
        #     state_next_seq = state_next_seq.cpu().detach()
        #     error = torch.mean((tmp_pred_s[0]-state_next_seq[0])**2)
        #     print(error)
        #     fig, ax = plt.subplots(1,2)
        #     ax[0].imshow(tmp_pred_s[0].numpy())
        #     ax[1].imshow(state_next_seq[0].numpy())
        #     plt.show()

            # latent_distance = torch.norm(z_seq[:, i, :] - z_next_seq[:, i, :], p=2, dim=-1)
            # original_distance = torch.norm(state_seq[:, i, :] - state_next_seq[:, i, :], p=2, dim=-1)
            # loss_distance += F.l1_loss(latent_distance, original_distance)

        return loss_fwd, loss_identity, self.C_forward  
    
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
    
    def save_C_forward(self, path, C_forward):
        C_forward_filename = path + '/' + 'C_forward.pt'
        print('[INFO] Saving C_forward weights to:', C_forward_filename)
        torch.save(C_forward, C_forward_filename)
        
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
        
        for i, batch_data in enumerate(dataloader):
            state, next_state = batch_data
            
            B = state.shape[0]
            
            state = state.to(device)
            next_state = next_state.to(device)
            state_feature = self.K_S(state)
            next_state_feature = self.K_S(next_state)
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
        plt.show()
        
        plt.imshow(B.cpu().detach().numpy())
        plt.colorbar()
        plt.show()
          
        if save_path is not None:
            save_path_Q = save_path + '/' + 'Q.pt'   
            print('[INFO] save Q to: ', save_path_Q)
            torch.save(Q, save_path_Q)
            
            save_path_B = save_path + '/' + 'B.pt'
            print('[INFO] save B to: ', save_path_B)
            torch.save(B, save_path_B)
            
            
        
        
    


class inverse_model(nn.Module):
     def __init__(self, K_O:nn.Module, K_H:nn.Module, K_S:nn.Module, K_S_preimage:nn.Module, direct_sum:bool=False, *args, **kwargs) -> None:
        super(inverse_model, self).__init__(*args, **kwargs)
        self.K_O = K_O
        self.K_H = K_H
        self.K_S = K_S
        self.K_S_preimage = K_S_preimage
        self.freeze_K_S()
        self.freeze_K_S_preimage()
        self.direct_sum = direct_sum
        
        if not direct_sum:
            self.C_S_OH = tltorch.TRL(input_shape=(K_H.hidden_dims[-1], K_O.hidden_dims[-1]), 
                                          output_shape=(K_S.hidden_dims[-1],))
        else: 
            self.C_S_OH = nn.Linear(K_H.hidden_dims[-1] + K_O.hidden_dims[-1], K_S.hidden_dims[-1])
     
     def forward(self, obs: Tensor, hist: Tensor):
         
        if not self.direct_sum:
            K_o = self.K_O(obs)
            K_h = self.K_H(hist)
            
            K_h = K_h.unsqueeze(2)
            K_o = K_o.unsqueeze(1)
            K_o_tensor_K_h = K_h * K_o
        else:
            K_o = self.K_O(obs)
            K_h = self.K_H(hist)
            K_o_tensor_K_h = torch.cat((K_o, K_h), dim=-1)
            
        K_s = self.C_S_OH(K_o_tensor_K_h)
        # state_analyzed = self.state_feature_decoder(state_feature_analyzed)
        # state_feature = self.state_feature_encoder(state_feature_analyzed)
        return K_s
    
     def compute_loss(self, hist: Tensor, obs: Tensor, state: Tensor):
         pred_K_s = self.forward(obs, hist)
         K_s = self.K_S(state)
         loss_1 = F.mse_loss(pred_K_s, K_s)
         loss_2 = F.mse_loss(self.K_S_preimage(pred_K_s), state)
         return loss_1, loss_2
    
     def freeze_K_S(self):
         print('[INFO] Freezing K_S')
         for param in self.K_S.parameters():
             param.requires_grad = False
    
     def defreeze_K_S(self):
         print('[INFO] Defreezing K_S')
         for param in self.K_S.parameters():
             param.requires_grad = True
             
     def freeze_K_S_preimage(self):
            print('[INFO] Freezing K_S_preimage')
            for param in self.K_S_preimage.parameters():
                param.requires_grad = False
    
     def defreeze_K_S_preimage(self):
            print('[INFO] Defreezing K_S_preimage')
            for param in self.K_S_preimage.parameters():
                param.requires_grad = True
    
     def save_model(self, path):
        self.to('cpu')
        if not self.direct_sum:
            model_path = path + '/' + 'inv_obs_model_tensor_product.pt'
            print('[INFO] Saving DA_Model to: ', model_path)
            torch.save(self.state_dict(), model_path)
        else:
            model_path = path + '/' + 'inv_obs_model_direct_sum.pt'
            print('[INFO] Saving DA_Model to: ', model_path)
            torch.save(self.state_dict(), model_path)


     def compute_R(self, da_dataset:torch.utils.data.Dataset, device:str="cpu", save_path:str=None):
         # R Compute the Covariance Matrix Cov(o_t,s_t)
         N = len(da_dataset)
         R = torch.zeros((N, self.K_S.hidden_dims[-1])).to(device)
         
         dataloader = torch.utils.data.DataLoader(da_dataset, batch_size=256, shuffle=False)
         
         for i, batch_data in enumerate(dataloader):
             obs, state, hist = batch_data
             obs = obs.to(device)
             state = state.to(device)
             hist = hist.to(device)
             Bs = hist.shape[0]
             K_s = self.K_S(state)
             pred_K_s = self.forward(obs=obs, hist=hist)
             obs_error = K_s - pred_K_s
             
             for j in range(Bs):
                 R[i*Bs+j] = obs_error[j]
                 
         R = torch.cov(R.T)    
         R = R.to("cpu")
         R = torch.pinverse(R + 1e-1*torch.eye(self.K_S.hidden_dims[-1]))
         
         
         if not is_symmetric(R):
            print('[INFO] R is not symmetric, using symmetriziation')
            R = 0.5*(R + R.T)
         else:
            print('[INFO] R is symmetric')

         plt.imshow(R.cpu().detach().numpy())
         plt.colorbar()
         plt.show()
          
          
         if save_path is not None: 
             save_path_R = save_path + '/' + 'R.pt' if not self.direct_sum else save_path + '/' + 'R_direct_sum.pt'  
             
             print('[INFO] save R to: ', save_path_R)    
             
             torch.save(R, save_path_R)
             
    

# Forward Operator C_{S_{t+1}|S_t}
class ERA5_forward_model(nn.Module):
    def __init__(self, K_S:nn.Module, K_S_preimage:nn.Module, 
                       seq_length: int, *args, **kwargs) -> None:
        super(ERA5_forward_model, self).__init__(*args, **kwargs)
        self.K_S = K_S
        self.K_S_preimage = K_S_preimage
        self.hidden_dim = K_S.hidden_dims[-1]
        self.seq_length = seq_length
        self.C_forward = None
    
    def forward(self, state: torch.Tensor):
        z, encode_list = self.K_S(state, return_encode_list=True)
        z_next = torch.mm(z, self.C_forward)
        pred_s_next = self.K_S_preimage(z_next, encode_list)
        return pred_s_next

    def batch_latent_forward(self, batch_z: torch.Tensor):
        B = batch_z.shape[0]
        if self.C_forward.dim() == 2:
            C_forward = self.C_forward.unsqueeze(0).repeat(B, 1, 1)
        else:
            C_forward = self.C_forward
        batch_z_next = torch.bmm(batch_z, C_forward)
        return batch_z_next

    def latent_forward(self, z: torch.Tensor):
        z_next = torch.mm(z, self.C_forward)
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
            z_seq[:, i, :], encode_z_step_t = self.K_S(state_seq[:, i, :], return_encode_list=True)
            encode_z_list.append(encode_z_step_t)

            z_next_seq[:, i, :] = self.K_S(state_next_seq[:, i, :])


        z_seq_pinv = self.batch_pinv(z_seq)
        forward_weights = torch.bmm(z_seq_pinv, z_next_seq).mean(dim=0).repeat(B, 1, 1)

        self.C_forward = forward_weights
        pred_z_next = self.batch_latent_forward(z_seq)
        
        
        for i in range(self.seq_length):
            recon_s = self.K_S_preimage(z_seq[:, i, :], encode_z_list[i])
            recon_s_next = self.K_S_preimage(pred_z_next[:, i, :], encode_z_list[i])

            # tmp_pred_s[:, i, :] = recon_s_next


            if weight_matrix is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, state_next_seq[:, i, :], weight_matrix).sum()
                loss_identity += weighted_MSELoss()(recon_s, state_seq[:, i, :], weight_matrix).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, state_next_seq[:, i, :])
                loss_identity += F.mse_loss(recon_s, state_seq[:, i, :])
                
        return loss_fwd, loss_identity, self.C_forward  
    
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
    
    def save_C_forward(self, path, C_forward):
        C_forward_filename = path + '/' + 'C_forward.pt'
        print('[INFO] Saving C_forward weights to:', C_forward_filename)
        torch.save(C_forward, C_forward_filename)
        
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
        
        for i, batch_data in enumerate(dataloader):
            state, next_state = batch_data
            
            B = state.shape[0]
            
            state = state.to(device)
            next_state = next_state.to(device)
            state_feature = self.K_S(state)
            next_state_feature = self.K_S(next_state)
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
        plt.show()
        
        plt.imshow(B.cpu().detach().numpy())
        plt.colorbar()
        plt.show()
          
        if save_path is not None:
            save_path_Q = save_path + '/' + 'Q.pt'   
            print('[INFO] save Q to: ', save_path_Q)
            torch.save(Q, save_path_Q)
            
            save_path_B = save_path + '/' + 'B.pt'
            print('[INFO] save B to: ', save_path_B)
            torch.save(B, save_path_B)
    def compute_z_b(self, dynamics_dataset:torch.utils.data.Dataset, device:str='cpu', save_path:str=None):
        N = len(dynamics_dataset)
        z_b = torch.zeros((N, self.hidden_dim)).to(device)
        BS = 32
        assert dynamics_dataset.seq_length == 1, "The sequence length of the dataset should be 1"
        
        dataloader = torch.utils.data.DataLoader(dynamics_dataset, batch_size=BS, shuffle=False)
        
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                state, _ = batch_data
                state = state.squeeze(1)
                bs = state.shape[0]
                state = state.to(device)
                state_feature = self.K_S(state)
                for j in range(bs):
                    z_b[i*bs+j] = state_feature[j]
        z_b = z_b.mean(dim=0)
        if save_path is not None:
            save_path_z_b = save_path + '/' + 'z_b.pt'
            print('[INFO] save z_b to: ', save_path_z_b)
            torch.save(z_b, save_path_z_b)


class ERA5_inverse_model(nn.Module):
    def __init__(self, K_O:nn.Module, K_S:nn.Module, K_S_preimage:nn.Module, *args, **kwargs) -> None:
        super(ERA5_inverse_model, self).__init__(*args, **kwargs)
        self.K_O = K_O
        self.K_S = K_S
        self.K_S_preimage = K_S_preimage
        self.freeze_K_S()
        self.freeze_K_S_preimage()
        
     
    def forward(self, obs: Tensor):
         
        
        # state_analyzed = self.state_feature_decoder(state_feature_analyzed)
        # state_feature = self.state_feature_encoder(state_feature_analyzed)
        return self.K_O(obs)
    
    def compute_loss(self, hist: Tensor, obs: Tensor, state: Tensor, weight_matrix=None):
        if weight_matrix is not None:
            pred_s = self.K_O(obs)
            loss = weighted_MSELoss()(pred_s, state, weight_matrix).sum()
        else:
            pred_s = self.K_O(obs)
            loss = F.mse_loss(pred_s, state)
        return loss
    
    def freeze_K_S(self):
         print('[INFO] Freezing K_S')
         for param in self.K_S.parameters():
             param.requires_grad = False
    
    def defreeze_K_S(self):
         print('[INFO] Defreezing K_S')
         for param in self.K_S.parameters():
             param.requires_grad = True
             
    def freeze_K_S_preimage(self):
            print('[INFO] Freezing K_S_preimage')
            for param in self.K_S_preimage.parameters():
                param.requires_grad = False
    
    def defreeze_K_S_preimage(self):
            print('[INFO] Defreezing K_S_preimage')
            for param in self.K_S_preimage.parameters():
                param.requires_grad = True
    
    def save_model(self, path):
        self.to('cpu')
        # if not self.direct_sum:
        #     model_path = path + '/' + 'inv_obs_model_tensor_product.pt'
        #     print('[INFO] Saving DA_Model to: ', model_path)
        #     torch.save(self.state_dict(), model_path)
        # else:
        model_path = path + '/' + 'inverse_model.pt'
        print('[INFO] Saving DA_Model to: ', model_path)
        torch.save(self.state_dict(), model_path)


    def compute_R(self, da_dataset:torch.utils.data.Dataset, device:str="cpu", save_path:str=None):
         # R Compute the Covariance Matrix Cov(o_t,s_t)
         N = len(da_dataset)
         R = torch.zeros((N, self.K_S.hidden_dims[-1])).to(device)
         
         dataloader = torch.utils.data.DataLoader(da_dataset, batch_size=256, shuffle=False)
         
         for i, batch_data in enumerate(dataloader):
             obs, state, hist = batch_data
             obs = obs.to(device)
             state = state.to(device)
             hist = hist.to(device)
             Bs = hist.shape[0]
             K_s = self.K_S(state)
             pred_K_s = self.forward(obs=obs, hist=hist)
             obs_error = K_s - pred_K_s
             
             for j in range(Bs):
                 R[i*Bs+j] = obs_error[j]
                 
         R = torch.cov(R.T)    
         R = R.to("cpu")
         R = torch.pinverse(R + 1e-1*torch.eye(self.K_S.hidden_dims[-1]))
         
         
         if not is_symmetric(R):
            print('[INFO] R is not symmetric, using symmetriziation')
            R = 0.5*(R + R.T)
         else:
            print('[INFO] R is symmetric')

         plt.imshow(R.cpu().detach().numpy())
         plt.colorbar()
         plt.show()
          
          
         if save_path is not None: 
             save_path_R = save_path + '/' + 'R.pt' if not self.direct_sum else save_path + '/' + 'R_direct_sum.pt'  
             
             print('[INFO] save R to: ', save_path_R)    
             
             torch.save(R, save_path_R)
             