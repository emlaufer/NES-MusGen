import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn
import os
import os.path
import pickle

PULSE1_THRESH = 31
PULSE2_THRESH = 31
TRI_THRESH = 20

PULSE1_NOTE_DIM = 108 - PULSE1_THRESH
PULSE2_NOTE_DIM = 108 - PULSE2_THRESH
TRI_NOTE_DIM = 108 - TRI_THRESH
NOISE_NOTE_DIM = 16

DIMENSIONS = [PULSE1_NOTE_DIM, 16, 4, PULSE2_NOTE_DIM, 16, 4, TRI_NOTE_DIM,
        NOISE_NOTE_DIM, 16, 2]
INDICES = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (3, 0), 
        (3, 1), (3, 2)]

if len(DIMENSIONS) != len(INDICES):
    raise ValueError('Not match dimensions and indices')

TOTAL_DIM = sum(DIMENSIONS) + 3 
SOS_IDX = TOTAL_DIM - 3
EOS_IDX = TOTAL_DIM - 2
NORMAL_IDX = TOTAL_DIM - 1

PAD_CATEGORIES = DIMENSIONS.copy()
PAD_CATEGORIES.append(3) # category for filler



def squeeze_categories(data):
    ''' Makes category numbers go from 0 to DIM '''

    for i in range(data.shape[0]):
        pulse1 = data[i][0][0] 
        if pulse1 != 0:
            if pulse1 <= PULSE1_THRESH:
                raise ValueError('Invalid pulse1 value')
            data[i][0][0] -= PULSE1_THRESH

        pulse2 = data[i][1][0] 
        if pulse2 != 0:
            if pulse2 <= PULSE2_THRESH:
                raise ValueError('Invalid pulse2 value')
            data[i][1][0] -= PULSE2_THRESH

        tri = data[i][2][0]
        if tri != 0:
            if tri <= TRI_THRESH:
                raise ValueError('Invalid triangle value')
            data[i][2][0] -= TRI_THRESH

def unsqueeze_categories(data):
    for i in range(data.shape[0]):
        if data[i][0][0] != 0:
            data[i][0][0] += PULSE1_THRESH

        if data[i][1][0] != 0:
            data[i][1][0] += PULSE2_THRESH

        if data[i][2][0] != 0:
            data[i][2][0] += TRI_THRESH


def exprsco_preprocess(data):
    ''' Preprocess raw numpy array '''
    data = data.astype(np.int)

    seq_len = data.shape[0]

    # (seq_len, dim)
    result = torch.zeros(seq_len + 2, TOTAL_DIM)
    note_targets = []

    sos_eos_targets = torch.full((seq_len + 2,), 2, dtype=torch.long)
    sos_eos_targets[0] = 0
    sos_eos_targets[-1] = 1
    
        
    squeeze_categories(data)

    # offset from the beginning of the array
    offset = 0
    r = np.arange(1, seq_len + 1)

    for dim, (idx1, idx2) in zip(DIMENSIONS, INDICES):
        note_targets.append(torch.empty(seq_len + 2, dtype=torch.long))

        note_data = data[:, idx1, idx2]
        torch_note_data = torch.from_numpy(note_data)
        note_targets[-1][1:-1] = torch_note_data
        note_data += offset

        result[r, note_data] = 1.

        note_targets[-1][0] = dim
        note_targets[-1][-1] = dim

        offset += dim


    result[1:seq_len + 1, NORMAL_IDX] = 1.

    result[0, SOS_IDX] = 1.
    result[result.shape[0] - 1, EOS_IDX] = 1.
    
    return (result, note_targets, sos_eos_targets)

def collate_expressive(minibatch):
    # TODO: needed?
    minibatch = sorted(minibatch, key=lambda x: x[0].shape[0], reverse=True)
    inputs, note_targets, sos_eos_targets = zip(*minibatch)
    note_targets = zip(*note_targets)

    lengths = [input.shape[0] for input in inputs]


    padded_inputs = rnn.pad_sequence(inputs, batch_first=True)

    padded_note_targets = [rnn.pad_sequence(target_list,
                padding_value=PAD_CATEGORIES[i], batch_first=True) 
            for i, target_list in enumerate(note_targets)] 

    padded_sos_eos_targets = rnn.pad_sequence(sos_eos_targets,
            padding_value=PAD_CATEGORIES[-1], batch_first=True)

    return padded_inputs, padded_note_targets, padded_sos_eos_targets, lengths


class NESExprDataset(Dataset): 
    def __init__(self, folder_name):
        self.folder_name = folder_name

        self.filenames = [os.path.join(folder_name, fn)
                for fn in os.listdir(folder_name)]
        fn = os.listdir(folder_name)[0]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        with open(self.filenames[index], 'rb') as f:
            rate, nsamps, exprsco = pickle.load(f)
            
        return exprsco_preprocess(exprsco)


def test_exprsco_preprocess(test_filename):
    print('Testing exprsco_preprocess')

    with open(test_filename, 'rb') as f:
        rate, nsamps, exprsco = pickle.load(f)

    processed = exprsco_preprocess(exprsco)
    print(processed.shape)
    print(processed[0, :])

    for i, v in enumerate(processed[0, :]):
        if v == 1.: print(i)


def main():
    test_filename = '../nesmdb24_exprsco/train/297_SkyKid_00_01StartMusic' + \
            'BGMIntroBGM.exprsco.pkl'
    if len(sys.argv) > 1: test_filename = sys.argv[1]


    test_exprsco_preprocess(test_filename)


if __name__ == '__main__':
    import sys
    main()
