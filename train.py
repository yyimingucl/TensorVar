from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
import os 
import torch
from utils import count_parameters

def train_inv_obs_model(inv_obs_model:Module, 
                        train_dataset:Dataset, 
                        model_save_folder:str, 
                        learning_rate:float=1e-3, 
                        nu:float=0.0, 
                        weight_decay:float=0, 
                        batch_size:int=64, 
                        num_epochs:int=20, 
                        gradclip:float=1,
                        device:str='cpu', 
                        with_consist:bool=False,
                        weight_matrix=None):
    print(f"[INFO] Training Inv_obs Model for {num_epochs} epochs")
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if with_consist:
        train_loss = {'latent match': [[] for _ in range(num_epochs)],
                    'original match': [[] for _ in range(num_epochs)],
                    'consist': [[] for _ in range(num_epochs)]}
    else:
        train_loss = {'latent match': [[] for _ in range(num_epochs)],
                    'original match': [[] for _ in range(num_epochs)]}
    
    inv_obs_model.freeze_phi_S()
    inv_obs_model.freeze_phi_inv_S()
    
    trainable_parameters = filter(lambda p: p.requires_grad, inv_obs_model.parameters())
    print(f"[INFO] Number of trainable parameters")
    count_parameters(inv_obs_model)
    
    optimizer = Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay) 
    inv_obs_model.train()
    inv_obs_model.to(device)

    loss_max = np.inf

    for epoch in range(num_epochs):
        all_loss = []
        for _, batch_data in tqdm(enumerate(train_dataloader), desc='Training Epochs {}'.format(epoch), total=len(train_dataloader)):
            obs, state, hist = [each_data.to(device) for each_data in batch_data]
            
            if weight_matrix is not None:
                B, C = batch_data[1].shape[0], batch_data[1].shape[1]
                weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                loss_1 = inv_obs_model.compute_loss(hist=hist, obs=obs, state=state, weight_matrix=weight_matrix_batch)
                loss_2 = torch.tensor(0)
            else:
                loss_1, loss_2 = inv_obs_model.compute_loss(hist=hist, obs=obs, state=state)
            loss = loss_1 + nu*loss_2
            train_loss['latent match'][epoch].append(loss_1.item())
            train_loss['original match'][epoch].append(loss_2.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(inv_obs_model.parameters(), gradclip) # gradient clip
            optimizer.step()       
            
            all_loss.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(all_loss):.4f}')
        print(f'latent match Loss: {np.mean(train_loss["latent match"][epoch]):.4f}')
        print(f'original match Loss: {np.mean(train_loss["original match"][epoch]):.4f}')

        if np.mean(all_loss) < loss_max:
            print(f"[INFO] Saving Inv obs model with loss: {np.mean(all_loss):.4f} at {model_save_folder}")
            loss_max = np.mean(all_loss)
            inv_obs_model.save_model(model_save_folder)
            inv_obs_model.to(device)

            
    # DA_Model.defreeze_state_encoder()
    inv_obs_model.defreeze_phi_S()
    inv_obs_model.defreeze_phi_inv_S()
    return train_loss


def train_forward_model(forward_model:Module, 
                        train_dataset:Dataset, 
                        model_save_folder:str, 
                        learning_rate:float=1e-3, 
                        lamb:float=0.3,  
                        weight_decay:float=0, 
                        batch_size:int=64, 
                        num_epochs:int=20, 
                        gradclip:float=1, 
                        device:str='cpu',
                        weight_matrix=None):
    

    print(f"[INFO] Training forward model for {num_epochs} epochs")
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_samples = len(train_dataset)
    
    train_loss = {'forward':[[] for _ in range(num_epochs)], 
                  'id':[[] for _ in range(num_epochs)]}

    trainable_parameters = filter(lambda p: p.requires_grad, forward_model.parameters())
    count_parameters(forward_model)

    
    optimizer = Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay) 

    forward_model.train()
    forward_model.to(device)

    loss_max = np.inf

    for epoch in range(num_epochs):
        all_loss = []
        C_fwd = torch.zeros((forward_model.hidden_dim, forward_model.hidden_dim), dtype=torch.float32).to(device)
        for _, batch_data in tqdm(enumerate(train_dataloader), desc='Training Epochs with lr={} :'.format(optimizer.param_groups[0]['lr']), total=len(train_dataloader)):
            
            B = batch_data[0].shape[0]
            if weight_matrix is not None:
                C = batch_data[0].shape[2]
                weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                pre_sequences, post_sequences = [each_data.to(device) for each_data in batch_data]
                loss_fwd, loss_id, tmp_C_fwd = forward_model.compute_loss(pre_sequences, post_sequences, weight_matrix_batch)
            else:
                pre_sequences, post_sequences = [each_data.to(device) for each_data in batch_data]
                loss_fwd, loss_id, tmp_C_fwd = forward_model.compute_loss(pre_sequences, post_sequences)
            
            loss = loss_fwd + lamb * loss_id 

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(forward_model.parameters(), gradclip) # gradient clip
            optimizer.step()       

            C_fwd += tmp_C_fwd*(B/num_samples)
            train_loss['forward'][epoch].append(loss_fwd.item())
            train_loss['id'][epoch].append(loss_id.item())
            all_loss.append(loss.item())

        train_loss['forward'][epoch] = np.mean(train_loss['forward'][epoch])
        train_loss['id'][epoch] = np.mean(train_loss['id'][epoch])


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(all_loss):.4f}')
        print(f'Forward Loss: {train_loss["forward"][epoch]:.4f}')
        print(f'ID Loss: {train_loss["id"][epoch]:.4f}')
        print('')


        if np.mean(all_loss) < loss_max:
            print(f"[INFO] Saving model with loss: {np.mean(all_loss):.4f} at {model_save_folder}")
            loss_max = np.mean(all_loss)
            forward_model.save_model(model_save_folder)
            forward_model.save_C_fwd(model_save_folder, C_fwd)
            forward_model.to(device)
    return train_loss

