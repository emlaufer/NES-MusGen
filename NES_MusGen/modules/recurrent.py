import torch
import torch.nn as nn
from ..data_process.expressive import TOTAL_DIM, DIMENSIONS

class BaselineLSTM(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.input_dim = TOTAL_DIM
        self.hidden_dim = hidden_dim
        self.output_dim = TOTAL_DIM

        # TODO: randomly initialize?
        # TODO: add learnable initial hidden states
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                num_layers=2, batch_first=True) 
        #nn.init.xavier_normal_(self.lstm.weight)

        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.xavier_normal_(self.fc1.weight)


    def forward(self, input, lengths):
        packed_in = nn.utils.rnn.pack_padded_sequence(input, lengths,
                batch_first=True)
        output, hidden = self.lstm(packed_in)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        out_logits = self.fc1(outputs)
        out_logits = out_logits.view(-1, self.output_dim)

        return out_logits


