import torch
import torch.nn as nn
import numpy as np


class MLN(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        """
        Args:
            x: (B, num_sample_tokens, C)
            c: (B, num_sample_tokens, 8), 8: fx, fy, x1, y1, z1, x2, y2, z2
        Returns:
            out: (B, num_sample_tokens, C)
        """
        x = self.ln(x)          # (B, num_sample_tokens, C)
        c = self.reduce(c)      # (B, num_sample_tokens, C)
        gamma = self.gamma(c)   # (B, num_sample_tokens, C)
        beta = self.beta(c)     # (B, num_sample_tokens, C)
        out = gamma * x + beta

        return out