import torch
import numpy as np
from torch.utils.data import Dataset
import os
import os.path

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

TOTAL_DIM = sum(DIMENSIONS)



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


def exprsco_preprocess(data):
    ''' Preprocess raw numpy array '''
    data = data.astype(np.int)

    seq_len = data.shape[0]

    # (seq_len, dim)
    result = torch.zeros(seq_len, TOTAL_DIM)
        
    squeeze_categories(data)

    # offset from the beginning of the array
    offset = 0
    r = np.arange(seq_len)

    for dim, (idx1, idx2) in zip(DIMENSIONS, INDICES):
        note_data = data[:, idx1, idx2]
        note_data += offset

        result[r, note_data] = 1.

        offset += dim

    return result

class NESExprDataset(Dataset): 
    def __init__(self, folder_name):
        self.folder_name = folder_name

        self.filenames = [os.path.realpath(fn) for fn in
                os.listdir(folder_name)]

    def __len__(self):
        return len(filenames)

    def __getitem__(self, index):
        with open(filenames[index], 'rb') as f:
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
    import pickle
    import sys
    main()
