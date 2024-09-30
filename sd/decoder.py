import torch 
from torch import nn
from torch.nn import functional as F
from attention import selfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = selfAttention(1, channels)

    def forward(self, x):
        #x: (batch_size, features, height, width)
        residue = x

        # (batch_size, features, height, width) -> (batch_size, features, height, width)
        x = self.groupnorm(x)

        n, c, h, w = x.shape

        # (batch_size, channels, height, width) -> (batch_size, channels, height*width)
        x = x.view((n, c, h*w))

        # (batch_size, chanels, height, width) -> (batch_size, channels, width, height)
        x = x.transpose(-1, -2)

        # perform self attention, no change in shape
        x = self.attention(x)

        #change the shape back to (batch_size, channels, height*width)
        x = x.transpose(-1, -2)

        #change it back to the original shape (batch_size, channels, height, width)
        x = x.view((n, c, h, w))

        #add the residual 
        x += residue

        return x

