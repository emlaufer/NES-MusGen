import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool
import numpy as np

from encoder import Encoder
from decoder import Decoder
from encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, depth=6, num_heads=8, input_dim=256, attn_dim=64, hidden_dim=2048):
        super(Transformer, self).__init__()

        self.embedings = nn.Embedding(vocab_size, input_dim)

        # add the positional encoding
        self.pos_enc = PositionalEncoding(input_dim, max_seq_len)

        # encoding and decoding layers to depth
        self.encoders = nn.ModuleList([Encoder(num_heads, input_dim, attn_dim, hidden_dim) for i in range(depth)])
        self.decoders = nn.ModuleList([Decoder(num_heads, input_dim, attn_dim, hidden_dim) for i in range(depth)])

        # final linear layer to output logits
        self.fc = nn.Linear(input_dim, vocab_size)

    def forward(self, x, y, src_mask, y_mask):
        x = self.embedings(x)
        x = self.pos_enc(x)

        for encoder in self.encoders:
            x = encoder(x, src_mask)

        for decoder in self.decoders:
            y = decoder(y, x, src_mask, y_mask)

        logits = self.fc(y)
        return logits

def test_transformer():
    print("-----Test Transformer Net-----")
    x_mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    y_mask = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=torch.float32)
    x = torch.tensor([0, 1, 2])
    y = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    model = Transformer(3, 3, input_dim=3)
    print(x)
    out = model(x, y, x_mask, y_mask)
    print(out)
    print(F.softmax(out, dim=1))

def main():
    test_transformer()

if __name__ == "__main__":
    test_transformer()
