import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool
import numpy as np

from attention import MultiHeadAttention

class Encoder(nn.Module):
    # output_dim must = input dim ?!
    def __init__(self, num_heads=8, input_dim=256, attn_dim=64, hidden_dim=2048):
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(num_heads, input_dim, hidden_dim, input_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x, mask=None):
        z = self.attention(x, x, x, mask)
        z = self.layer_norm1(x + z)
        # TODO add dropout
        out = self.fc2(F.relu(self.fc1(z)))
        out = self.layer_norm2(out + z)
        return out

def test_encoder():
    print("-----Testing Encoder-----")
    enc = Encoder(8, 3, 2)
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    print(a)
    print(a.shape)
    out = enc(a)
    print(out)
    print(out.shape)

def main():
    test_encoder()

if __name__ == "__main__":
    main()
