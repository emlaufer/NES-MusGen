import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool
import numpy as np

from attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, num_heads=8, input_dim=256, attn_dim=64, hidden_dim=2048):
        super(Decoder, self).__init__()

        self.self_attention = MultiHeadAttention(num_heads, input_dim, hidden_dim, input_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.enc_dec_attention = MultiHeadAttention(num_heads, input_dim, hidden_dim, input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)

    def forward(self, y, encoder_output, enc_mask=None, y_mask=None):
        z = self.self_attention(y, y, y, y_mask)
        z = self.layer_norm1(y + z)
        z2 = self.enc_dec_attention(z, encoder_output, encoder_output, enc_mask)
        z2 = self.layer_norm2(z2 + z)
        out = self.fc2(F.relu(self.fc1(z2)))
        out = self.layer_norm2(out + z2)
        return out
