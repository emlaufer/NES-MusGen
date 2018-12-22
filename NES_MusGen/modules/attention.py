import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool
import numpy as np

class ScaledDotAttention(nn.Module):
    def __init__(self, in_dim=256, out_dim=64):
        super(ScaledDotAttention, self).__init__()

        # used for scaling
        self.sqrt_out_dim = np.sqrt(out_dim)

        # query, key, and value weight matrices
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)

    # in self-attention, we use x for all these
    # for encoder-decoder attn, we use x, encoder_out, encoder_out
    def forward(self, q, k, v, mask=None):
        # calculate the query, key, and values
        query = self.q(q)
        key = self.k(k)
        value = self.v(v)
  
        # score = (query x key^T) / sqrt_dim
        scores = torch.matmul(query, torch.t(key)) / self.sqrt_out_dim

        # replace masked values with min float
        if mask is not None:
            scores[mask == 0] = -1 * torch.finfo(scores.dtype).max

        # softmax accross score rows and multiply by value matrix
        # at input i, the softmax[i] = the percent to attend to each input
        output = torch.matmul(F.softmax(scores, dim=1), value)

        return output

class MultiHeadAttention(nn.Module):
    # will multiprocessing speed this up?? is it worth it?
    def __init__(self, num_heads=8, in_dim=256, hidden_dim=64, out_dim=256):
        super(MultiHeadAttention, self).__init__()

        #self.in_dim = in_dim
        #self.out_
        self.hidden_dim = hidden_dim

        self.heads = nn.ModuleList([ScaledDotAttention(in_dim, hidden_dim) for i in range(num_heads)])
        self.linear = nn.Linear(hidden_dim * num_heads, out_dim)

    # in self-attention, we use x for all these
    # for encoder-decoder attn, we use x, encoder_out, encoder_out
    def forward(self, q, k, v, mask=None):
        # create the output tensor of shape B x (D_h * num_heads)
        # should shape use q, or k, or v?
        cat = torch.zeros(q.shape[0], len(self.heads) * self.hidden_dim, dtype=torch.float32)

        # run the input through each head
        for i, head in enumerate(self.heads):
            # concate each output into out output vector
            cat[:, i*self.hidden_dim:(i+1)*self.hidden_dim] = head(q, k, v, mask)

        # lastly, run it through our linear layer
        out = self.linear(cat)
        return out
  
def test_scaled_attn():
    print("-----Testing Scaled Attn-----")
    attn = ScaledDotAttention(3, 3)
    #a = torch.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9]], dtype=torch.float32)
    a = torch.tensor([[1, 2, 3], [4, 5 , 6]], dtype=torch.float32)
    print(a)
    print(a.shape)
    out = attn(a, a, a)
    print(out)
    print(out.shape)

def test_multi_attn():
    print("-----Testing MultiHead Attn-----")
    attn = MultiHeadAttention(2, 3, 2, 3)
    #a = torch.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9]], dtype=torch.float32)
    a = torch.tensor([[1, 2, 3], [4, 5 , 6]], dtype=torch.float32)
    print(a)
    out = attn(a, a, a)
    print(out)

def test_masked_attn():
    print("-----Testing MultiHead Attn with Masking-----")
    attn = MultiHeadAttention(2, 3, 2, 3)
    #a = torch.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9]], dtype=torch.float32)
    a = torch.tensor([[1, 2, 3], [4, 5 , 6]], dtype=torch.float32)
    print(a)
    mask = torch.tensor([[1, 0], [1, 1]])
    print(mask)
    out = attn(a, a, a, mask=mask)
    print(out)


def main():
    test_scaled_attn()
    test_multi_attn()
    test_masked_attn()

if __name__ == "__main__":
    main()
