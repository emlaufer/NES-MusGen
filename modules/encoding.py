import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, dimensions=256, max_seq_len=256):
        super(PositionalEncoding, self).__init__()
        PosEnc = torch.zeros(max_seq_len, dimensions)

        for i in range(max_seq_len):
            for j in range(dimensions):
                if j % 2 == 0:
                    PosEnc[i, j] = np.sin(i / (np.power(10000, 2 * j / dimensions)))
                else:
                    PosEnc[i, j] = np.cos(i / np.power(10000, 2 * j / dimensions))

        # regsters the Positive offset as a persistent buffer
        self.register_buffer('PosEnc', PosEnc)

    def forward(self, x):
        # add x (B x D) to PosEnc (MB x D)
        return x + self.PosEnc[:x.shape[0], :]

def test_pos_enc():
    print("-----Testing Pos Enc-----")
    enc = PositionalEncoding(3, 3)
    a = torch.tensor([[1, 2, 3], [4, 5 , 6]], dtype=torch.float32)
    print(a)
    print(a.shape)
    out = enc(a)
    print(out)
    print(out.shape)

def main():
    test_pos_enc()

if __name__ == "__main__":
    main()
