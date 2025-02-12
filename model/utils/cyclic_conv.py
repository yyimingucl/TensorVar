from torch import nn
from torch.nn import functional as F


class conv_layer_circular_padding(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, *args, **kwargs):
        super(conv_layer_circular_padding, self).__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_c, out_c, 
                              kernel_size=kernel_size, 
                              padding=0,
                              stride=stride,
                              *args, **kwargs)
    def forward(self, x):
        x = F.pad(x,pad=(self.kernel_size // 2, 
                         self.kernel_size // 2,
                         self.kernel_size // 2,
                         self.kernel_size // 2),
                         mode='circular')
        return self.conv(x)
    
class convtranspose_layer_circular_padding(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, out_padding=0, *args, **kwargs):
        super(convtranspose_layer_circular_padding, self).__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.conv = nn.ConvTranspose2d(in_c, out_c, 
                                       kernel_size=kernel_size, 
                                       stride=stride, 
                                       padding=0,
                                       output_padding=out_padding,
                                       *args, **kwargs)
    def forward(self, x):
        pad_size = self.kernel_size // 2
        x = F.pad(x,pad=(pad_size, pad_size, pad_size, pad_size),
                    mode='circular')
        x = self.conv(x)
        return x[..., self.kernel_size // 2:-self.kernel_size // 2, self.kernel_size // 2:-self.kernel_size // 2]