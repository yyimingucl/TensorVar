from torch.nn.modules import Module
from torch import nn
import torch
from .utils.utils import Transformer_Based_Inv_Obs_Model
from .utils.ERA5_base import *

'''
================================
NN features for Climate system
================================
'''


class ERA5_phi_O(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(ERA5_phi_O, self).__init__(*args, **kwargs)
        self.input_dim = config.obs_dim[0] * config.history_len
        self.output_dim = config.state_dim[0]

        self.features = Transformer_Based_Inv_Obs_Model(in_channel=self.input_dim, out_channel=self.output_dim)

    def forward(self, obs: torch.Tensor):
        return self.features(obs)


class ERA5_phi_S(Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(ERA5_phi_S, self).__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = config.state_dim
        self.filter_dims = config.state_filter_feature_dim

        self.Conv2D_size5_1 = nn.Conv2d(in_channels=self.input_dim, out_channels=self.filter_dims[0], 
                                  kernel_size=9, stride=1, padding=4)
        
        self.Conv2D_size5_2 =  nn.Conv2d(in_channels=self.filter_dims[0], out_channels=self.filter_dims[0], 
                                  kernel_size=5, stride=1, padding=2)
        
        self.Conv2D_size3_1 = nn.Conv2d(in_channels=self.filter_dims[0], out_channels=self.filter_dims[1], 
                                              kernel_size=3, stride=1, padding=1)
        
        self.Conv2D_size3_2 = nn.Conv2d(in_channels=self.filter_dims[1], out_channels=self.filter_dims[2], 
                                              kernel_size=3, stride=1, padding=1)
        
        self.Conv2D_size3_3 = nn.Conv2d(in_channels=self.filter_dims[2], out_channels=self.filter_dims[3], 
                                              kernel_size=3, stride=1, padding=1)

        self.flat_dim = int(config.state_filter_feature_dim[-1] * (config.state_dim[1] * config.state_dim[2]) / 256)
        self.feature_dim = config.feature_dim
        self.flatten = nn.Flatten()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.flat_dim, self.feature_dim)


    def forward(self, state: torch.Tensor, return_encode_list=False):
        if return_encode_list:
            encode_list = [state.clone()]

            en_state_1 = self.Conv2D_size5_1(state)
            encode_list.append(en_state_1.clone())
            en_state_1 = self.pooling(en_state_1)
            en_state_1 = self.relu(en_state_1)

            en_state_2 = self.Conv2D_size5_2(en_state_1)
            encode_list.append(en_state_2.clone())
            en_state_2 = self.relu(en_state_2)
            en_state_2 = self.pooling(en_state_2)

            en_state_3 = self.Conv2D_size3_1(en_state_2)
            encode_list.append(en_state_3.clone())
            en_state_3 = self.pooling(en_state_3)
            en_state_3 = self.relu(en_state_3)

            en_state_4 = self.Conv2D_size3_2(en_state_3)
            encode_list.append(en_state_4.clone())
            en_state_4 = self.pooling(en_state_4)
            en_state_4 = self.relu(en_state_4)

            en_state_5 = self.Conv2D_size3_3(en_state_4)
            encode_list.append(en_state_5.clone())
            en_state_5 = self.relu(en_state_5)

            en_state_5 = self.flatten(en_state_5)
            z = self.linear(en_state_5)

            return z, encode_list
        else:
            en_state_1 = self.Conv2D_size5_1(state)
            en_state_1 = self.pooling(en_state_1)
            en_state_1 = self.relu(en_state_1)

            en_state_2 = self.Conv2D_size5_2(en_state_1)
            en_state_2 = self.relu(en_state_2)
            en_state_2 = self.pooling(en_state_2)

            en_state_3 = self.Conv2D_size3_1(en_state_2)
            en_state_3 = self.pooling(en_state_3)
            en_state_3 = self.relu(en_state_3)

            en_state_4 = self.Conv2D_size3_2(en_state_3)
            en_state_4 = self.pooling(en_state_4)
            en_state_4 = self.relu(en_state_4)

            en_state_5 = self.Conv2D_size3_3(en_state_4)
            en_state_5 = self.relu(en_state_5)

            en_state_5 = self.flatten(en_state_5)

            z = self.linear(en_state_5)

            return z



class ERA5_phi_inv_S(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(ERA5_phi_inv_S, self).__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = config.state_dim
        self.filter_dims = config.state_filter_feature_dim
        self.flat_dim = int(config.state_filter_feature_dim[-1] * (config.state_dim[1] * config.state_dim[2]) / 256)
        self.feature_dim= config.feature_dim

        self.linear = nn.Linear(self.feature_dim, self.flat_dim)
        
        self.ConvTranspose2D_size3_1 = nn.ConvTranspose2d(in_channels=self.filter_dims[3], out_channels=self.filter_dims[2], 
                                                          kernel_size=3, stride=1, padding=1)
        self.ConvTranspose2D_size3_2 = nn.ConvTranspose2d(in_channels=self.filter_dims[2], out_channels=self.filter_dims[1],
                                                          kernel_size=3, stride=1, padding=1)
        self.ConvTranspose2D_size3_3 = nn.ConvTranspose2d(in_channels=self.filter_dims[1], out_channels=self.filter_dims[0],
                                                          kernel_size=3, stride=1, padding=1)
        
        self.Upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ConvTranspose2D_size5_1 = nn.ConvTranspose2d(in_channels=self.filter_dims[0], out_channels=self.filter_dims[0],
                                                          kernel_size=5, stride=1, padding=2)
        
        self.ConvTranspose2D_size5_2 = nn.ConvTranspose2d(in_channels=self.filter_dims[0], out_channels=self.input_dim,
                                                          kernel_size=9, stride=1, padding=4)
        
        self.relu = nn.ReLU()
        self.output_conv = nn.Sequential(nn.Conv2d(in_channels=self.input_dim, out_channels=128, 
                                         kernel_size=1, stride=1),
                                         nn.ReLU(),

                                         nn.Conv2d(in_channels=128, out_channels=self.input_dim, 
                                         kernel_size=1, stride=1))



    def forward(self, z: torch.Tensor, encode_list:list[torch.Tensor]=None):
        if encode_list is not None:

            de_state_5 = self.linear(z)
            de_state_5 = self.relu(de_state_5)
            de_state_5 = de_state_5.view(-1, self.filter_dims[3], self.w//16, self.h//16)


            de_state_4 = self.ConvTranspose2D_size3_1(de_state_5+encode_list[-1])
            de_state_4 = self.relu(de_state_4)

            de_state_3 = self.Upsampling(de_state_4)
            de_state_3 = self.ConvTranspose2D_size3_2(de_state_3+encode_list[-2])
            de_state_3 = self.relu(de_state_3)

            de_state_2 = self.Upsampling(de_state_3) 
            de_state_2 = self.ConvTranspose2D_size3_3(de_state_2+encode_list[-3])
            de_state_2 = self.relu(de_state_2)

            de_state_1 = self.Upsampling(de_state_2)
            de_state_1 = self.ConvTranspose2D_size5_1(de_state_1+encode_list[-4])
            de_state_1 = self.relu(de_state_1)

            de_state_0 = self.Upsampling(de_state_1)
            de_state_0 = self.ConvTranspose2D_size5_2(de_state_0+encode_list[-5])
            recon_s = self.output_conv(de_state_0)

            return recon_s
        
        else:
            de_state_5 = self.linear(z)
            de_state_5 = self.relu(de_state_5)
            de_state_5 = de_state_5.view(-1, self.filter_dims[3], self.w//16, self.h//16)

            de_state_4 = self.ConvTranspose2D_size3_1(de_state_5)
            de_state_4 = self.relu(de_state_4)

            de_state_3 = self.Upsampling(de_state_4)
            de_state_3 = self.ConvTranspose2D_size3_2(de_state_4)
            de_state_3 = self.relu(de_state_3)

            de_state_2 = self.Upsampling(de_state_3) 
            de_state_2 = self.ConvTranspose2D_size3_3(de_state_3)
            de_state_2 = self.relu(de_state_2)

            de_state_1 = self.Upsampling(de_state_2)
            de_state_1 = self.ConvTranspose2D_size5_1(de_state_1)
            de_state_1 = self.relu(de_state_1)

            de_state_0 = self.Upsampling(de_state_1)
            de_state_0 = self.ConvTranspose2D_size5_2(de_state_1)
            recon_s = self.output_conv(de_state_0)

            return recon_s
'''
=============================
Operators for ERA 5 system
=============================
'''

class ERA5_forward_model(forward_model):
    def __init__(self, config, *args, **kwargs) -> None:
        phi_S = ERA5_phi_S(config)
        phi_inv_S = ERA5_phi_inv_S(config)
        seq_length = config.seq_length
        super(ERA5_forward_model, self).__init__(phi_S=phi_S,
                                            phi_inv_S=phi_inv_S, 
                                            seq_length=seq_length,
                                            *args, **kwargs)

class ERA5_inverse_model(inverse_model):
    def __init__(self, config, *args, **kwargs) -> None:
        phi_O = ERA5_phi_O(config)
        phi_S = ERA5_phi_S(config)
        phi_inv_S = ERA5_phi_inv_S(config)
        super(ERA5_inverse_model, self).__init__(phi_O=phi_O,
                                             phi_S=phi_S,
                                             phi_inv_S=phi_inv_S, 
                                             *args, **kwargs)

