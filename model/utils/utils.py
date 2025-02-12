from torch.nn import Module
from torch import Tensor, FloatTensor, pow, sin, cos, arange
import torch


class View(Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class PositionalEncodingLayer(Module):
    
    def __init__(self, input_dim: int, max_len: int=100):
        super(PositionalEncodingLayer, self).__init__()
        self.input_dim = input_dim
        self.max_len = max_len
    
    def get_angles(self, positions: Tensor, indexes: Tensor):
        input_dim_tensor = FloatTensor([[self.input_dim]]).to(positions.device)
        angle_rates = pow(10000, (2 * (indexes // 2)) / input_dim_tensor)
        return positions / angle_rates

    def forward(self, input_sequences: Tensor, channel_first: bool=False):
        """
        :param Tensor[batch_size, seq_len] input_sequences
        :return Tensor[batch_size, seq_len, input_dim] position_encoding
        """
        assert len(input_sequences.shape) == 3, "input_sequences must be of shape [batch_size, seq_len, input_dim]"
        if channel_first:
            input_sequences = input_sequences.permute(0, 2, 1)
        positions = arange(input_sequences.size(1)).unsqueeze(1).to(input_sequences.device) # [seq_len, 1]
        indexes = arange(self.input_dim).unsqueeze(0).to(input_sequences.device) # [1, input_dim]
        angles = self.get_angles(positions, indexes) # [seq_len, input_dim]
        angles[:, 0::2] = sin(angles[:, 0::2]) # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = cos(angles[:, 1::2]) # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(input_sequences.size(0), 1, 1) # [batch_size, seq_len, input_dim]
        if channel_first:
            position_encoding = position_encoding.permute(0, 2, 1)
        return position_encoding


def is_symmetric(matrix, tol=1e-8):
    return torch.allclose(matrix, matrix.T, atol=tol)  


