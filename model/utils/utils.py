from torch.nn import Module
from torch import Tensor, FloatTensor, pow, sin, cos, arange
import torch
from .transformer import *


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



class TimeEmbedding(nn.Module):
    def __init__(self, in_c: int, t_len: int, out_dim: int, embed_dim: int = 32):
        """
        Time Embedding Module.
        - Uses sine-cosine positional encoding for time representation.
        - Applies convolutional projections to mix temporal and spatial information.

        Args:
            in_c (int): Number of input channels.
            t_len (int): Number of time steps.
            out_dim (int): Output embedding dimension.
            embed_dim (int): Dimension for time embeddings.
        """
        super(TimeEmbedding, self).__init__()
        self.in_c = in_c
        self.t_len = t_len
        self.out_dim = out_dim
        self.embed_dim = embed_dim

        # Learnable time embedding layer
        self.time_embedding = nn.Linear(embed_dim, in_c)

        # Convolutional layers for projection
        self.proj_1 = conv_layer_circular_padding(in_c, out_dim // 2, kernel_size=3)
        self.proj_2 = conv_layer_circular_padding(out_dim // 2, out_dim, kernel_size=3)

        self.time_out = nn.Conv1d(t_len, 1, kernel_size=1, padding=0)

        self.norm = nn.LayerNorm([in_c, t_len, 1, 1])  # Normalize time dimension

    def fourier_time_embedding(self, T, device):
        """
        Compute sine-cosine positional encoding for time representation.
        Args:
            T (int): Number of time steps.
            device: Device to allocate tensor.
        Returns:
            Tensor: Time embeddings (T, embed_dim).
        """
        t = torch.arange(T, device=device).float().unsqueeze(1)  # (T, 1)
        freqs = torch.exp(-torch.arange(0, self.embed_dim, 2, device=device).float() * (torch.log(torch.tensor(10000.0)) / self.embed_dim))
        time_emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)  # (T, embed_dim)
        return time_emb

    def forward(self, x):
        """
        Forward pass for time embedding.
        Args:
            x (Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            Tensor: Processed output of shape (B, out_dim, H, W)
        """
        B, T, C, H, W = x.shape

        # Generate and apply time embeddings
        time_emb = self.fourier_time_embedding(T, x.device)  # (T, embed_dim)
        time_emb = self.time_embedding(time_emb).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, T, C, 1, 1)
        x = x + time_emb  # Inject time encoding

        # Reshape and project
        x = x.view(B * T, C, H, W)  # Merge batch and time for convolution
        x = F.relu(self.proj_1(x))
        x = self.proj_2(x)

        # Reshape back to batch form
        x = x.view(B, T, self.out_dim*H, W)
        x = self.time_out(x)
        x = x.view(B, self.out_dim, H, W)

        return x

class weighted_MSELoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return ((inputs - targets)**2 ) * weights       



class Transformer_Based_Inv_Obs_Model(nn.Module):
    def __init__(self, in_channel:int=50, out_channel:int=5, LayerNorm_type = 'WithBias',
                 ffn_expansion_factor = 2.66, bias = False, num_blocks = [1, 1, 2, 2]):
        super(Transformer_Based_Inv_Obs_Model, self).__init__()

        dim_list = [in_channel*2, in_channel, in_channel // 2, out_channel]
        num_heads = [5, 10, 5, 1]
        num_blocks = num_blocks

        self.patch_embed = OverlapPatchEmbed(in_channel, embed_dim=dim_list[0])
        self.Upsample_1 = Flex_Sample(dim_list[0], dim_list[1])
        self.Upsample_2 = Flex_Sample(dim_list[1], dim_list[2])
        self.Upsample_3 = Flex_Sample(dim_list[2], dim_list[3])

        self.block1 = nn.Sequential(*[TransformerBlock(dim=dim_list[0], 
                                                       num_heads=num_heads[0], 
                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                       bias=bias, LayerNorm_type=LayerNorm_type) 
                                                       for i in range(num_blocks[0])])
        self.block2 = nn.Sequential(*[TransformerBlock(dim=dim_list[1], 
                                                       num_heads=num_heads[1], 
                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                       bias=bias, LayerNorm_type=LayerNorm_type) 
                                                       for i in range(num_blocks[1])])
        self.block3 = nn.Sequential(*[TransformerBlock(dim=dim_list[2],
                                                         num_heads=num_heads[2], 
                                                         ffn_expansion_factor=ffn_expansion_factor, 
                                                         bias=bias, LayerNorm_type=LayerNorm_type) 
                                                         for i in range(num_blocks[2])])
        self.block4 = nn.Sequential(*[TransformerBlock(dim=dim_list[3],
                                                        num_heads=num_heads[3], 
                                                        ffn_expansion_factor=ffn_expansion_factor, 
                                                        bias=bias, LayerNorm_type=LayerNorm_type) 
                                                        for i in range(num_blocks[3])])
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.block1(x)
        x = self.Upsample_1(x)

        x = self.block2(x)
        x = self.Upsample_2(x)

        x = self.block3(x)
        x = self.Upsample_3(x)

        x = self.block4(x)
        return x


        

