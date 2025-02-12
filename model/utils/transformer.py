import torch
import torch.nn as nn
import torch.nn.functional as F
from .cyclic_conv import *
import numbers

from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
class BiasFree_LayerNorm1D(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm1D, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class WithBias_LayerNorm1D(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm1D, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        B, _, L = x.size()
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        weight = self.weight.repeat(B,1).unsqueeze(-1).repeat(1,1,L)
        bias = self.bias.repeat(B,1).unsqueeze(-1).repeat(1,1,L)
        return (x - mu) / torch.sqrt(sigma+1e-5) * weight + bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    
class LayerNorm1D(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm1D, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm1D(dim)
        else:
            self.body = WithBias_LayerNorm1D(dim)

    def forward(self, x):
        return self.body(x)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = conv_layer_circular_padding(dim, hidden_features*2, kernel_size=1, stride=1)

        self.dwconv = conv_layer_circular_padding(hidden_features*2, hidden_features*2, kernel_size=3, stride=1)

        self.project_out = conv_layer_circular_padding(hidden_features, dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.tanh(x1) * x2
        x = self.project_out(x)
        return x

class FeedForward1D(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward1D, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv1d(dim, hidden_features*2, kernel_size=1)

        self.dwconv = nn.Conv1d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2)

        self.project_out = nn.Conv1d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.tanh(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = conv_layer_circular_padding(dim, dim*3, kernel_size=1, stride=1)
        self.qkv_dwconv = conv_layer_circular_padding(dim*3, dim*3, kernel_size=3, stride=1)
        self.project_out = conv_layer_circular_padding(dim, dim, kernel_size=1, stride=1,)


    def forward(self, x):
        _,_,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = conv_layer_circular_padding(in_c, embed_dim, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.proj(x)

        return x




class Flex_Sample(nn.Module):
    def __init__(self, in_channels, out_channels, up=True):
        super(Flex_Sample, self).__init__()
        if up:
            self.body = conv_layer_circular_padding(in_channels, 
                                                    out_channels, 
                                                    kernel_size=3, 
                                                    stride=1)
        else:
            self.body = convtranspose_layer_circular_padding(in_channels, 
                                                             out_channels, 
                                                             kernel_size=3, 
                                                             stride=1, 
                                                             out_padding=0)
    def forward(self, x):
        return self.body(x)