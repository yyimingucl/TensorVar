from .base import *
from .utils.utils import PositionalEncodingLayer

'''
================================
Features for Kuramoto-Sivashinsky system
================================
'''

class KS_phi_O(phi_O_BASE):
    def __init__(self, config, *args, **kwargs) -> None:
        self.input_dim = config.obs_dim
        self.hidden_dims = config.obs_feature_dim

        features = nn.ModuleList()
        for i in range(len(self.hidden_dims)):
            if i == 0:
                features.append(nn.Linear(self.input_dim, self.hidden_dims[i]))
                features.append(nn.ReLU())
            else:
                features.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
                features.append(nn.ReLU())
        features = nn.Sequential(*features)
        super(KS_phi_O, self).__init__(features, *args, **kwargs)
    


class KS_hist_1dcov_features(nn.Module):
    def __init__(self, 
                 input_dim:int, 
                 history_len:int, 
                 hidden_dims:list, 
                 cov_kernel_size:list, 
                 pooling_size:int, 
                 padding:list, 
                 position_encoding:bool=True, *args, **kwargs) -> None:
        super(KS_hist_1dcov_features, self).__init__(*args, **kwargs)
        assert len(cov_kernel_size) == len(hidden_dims), "The length of cov_kernel_size should be the same as the length of hidden_dims"
        self.input_dim = input_dim
        self.history_len = history_len
        self.hidden_dims = hidden_dims
        self.pooling_size = pooling_size
                 
        L_out = history_len
        padding = padding
        dilation=1
        stride=1
        
        if position_encoding:
            self.position_encoding = PositionalEncodingLayer(input_dim=input_dim)
            print('[INFO] Positional Encoding is used')
        else:
            self.position_encoding = None
            print('[INFO] Positional Encoding is not used')
        
        
        for i in range(len(hidden_dims)):
            if i == 0:
                if position_encoding:
                    self.conv1 = nn.Conv1d(input_dim*2, hidden_dims[i], cov_kernel_size[i], padding=padding[i])
                else:
                    self.conv1 = nn.Conv1d(input_dim, hidden_dims[i], cov_kernel_size[i], padding=padding[i])
            else:
                setattr(self, f'conv{i+1}', nn.Conv1d(hidden_dims[i-1], hidden_dims[i], cov_kernel_size[i], padding=padding[i]))
            setattr(self, f'pool{i+1}', nn.MaxPool1d(pooling_size, stride=stride))
            
            L_out = int((L_out + 2*padding[i] - (dilation*cov_kernel_size[i]-1) -1)/stride)
        
        self.flatten = nn.Flatten()
        self.flatten_size = L_out * hidden_dims[-1]
        print('[INFO] Input size of the fully connected layer: ', self.flatten_size)


        self.relu = nn.ReLU()
        self.nn = nn.Linear(self.flatten_size, hidden_dims[-1])
    
    def forward(self, x: torch.Tensor):
        hidden_dims = self.hidden_dims
        if self.position_encoding is not None:
            x = torch.concat([x, self.position_encoding(x, channel_first=True)], dim=1)
        for i in range(len(hidden_dims)):
            if i == 0:
                x = self.conv1(x)
            else:
                x = getattr(self, f'conv{i+1}')(x)
            x = getattr(self, f'pool{i+1}')(x)
            x = self.relu(x)
        x = self.flatten(x)
        x = self.nn(x)
        return x    


class KS_phi_H(phi_H_BASE):
    def __init__(self, config, *args, **kwargs) -> None:
        self.input_dim = config.obs_dim
        self.history_len = config.history_len
        self.hidden_dims = config.hist_feature_dim
        self.cov_kernel_size = config.cov_kernel_size # [5,3,3]
        self.padding = config.padding # [2, 1, 1]
        self.pooling_size = config.pooling_size # 2
        self.position_encoding = config.position_encoding
        features = KS_hist_1dcov_features(input_dim=self.input_dim, 
                                           history_len=self.history_len,
                                           hidden_dims=self.hidden_dims, 
                                           cov_kernel_size=self.cov_kernel_size, 
                                           pooling_size=self.pooling_size, 
                                           padding=self.padding,
                                           position_encoding=self.position_encoding)
        super(KS_phi_H, self).__init__(features, hist_w=self.history_len, *args, **kwargs)


class KS_phi_S(phi_S_BASE):
    def __init__(self, config, *args, **kwargs) -> None:
        self.input_dim = config.state_dim
        self.hidden_dims = config.state_feature_dim
        features = nn.ModuleList()
        for i in range(len(self.hidden_dims)):
            if i == 0:
                features.append(nn.Linear(self.input_dim, self.hidden_dims[i]))
                features.append(nn.ReLU())
            else:
                features.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
                if i < len(self.hidden_dims) - 1:
                    features.append(nn.ReLU())
        features = nn.Sequential(*features)
        super(KS_phi_S, self).__init__(features, *args, **kwargs)

class KS_phi_inv_S(phi_inv_S_BASE):
    def __init__(self, config, *args, **kwargs) -> None:
        self.input_dim = config.state_dim
        self.hidden_dims = config.state_feature_dim
        features = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            if i == len(self.hidden_dims) - 1:
                features.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i - 1]))
            else:
                features.append(nn.ReLU())
                features.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i - 1] if i > 0 else self.input_dim))
        features = nn.Sequential(*features)
        super(KS_phi_inv_S, self).__init__(features, *args, **kwargs)
    
'''
=============================
models for Kuramoto-Sivashinsky system
=============================
'''

class KS_foward_model(forward_model):
    def __init__(self, config, *args, **kwargs) -> None:
        phi_S = KS_phi_S(config)
        phi_inv_S = KS_phi_inv_S(config)
        seq_length = config.seq_length
        super(KS_foward_model, self).__init__(phi_S=phi_S,
                                               phi_inv_S=phi_inv_S, 
                                               seq_length=seq_length,
                                               *args, **kwargs)
        
class KS_inv_obs_model(inverse_model):
    def __init__(self, config, *args, **kwargs) -> None:
        phi_O = KS_phi_O(config)
        phi_H = KS_phi_H(config)
        phi_S = KS_phi_S(config)
        phi_inv_S = KS_phi_inv_S(config)
        rate = config.rate
        super(KS_inv_obs_model, self).__init__(phi_O=phi_O, 
                                                phi_H=phi_H, 
                                                phi_S=phi_S, 
                                                phi_inv_S=phi_inv_S,
                                                rate=rate, 
                                                *args, **kwargs)
        
        

