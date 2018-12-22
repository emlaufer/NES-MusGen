from ..data_process.expressive import NESExprDataset, collate_expressive, \
        DIMENSIONS, TOTAL_DIM
from ..modules import BaselineLSTM
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import numpy as np
import os.path

DEFAULT_CONFIG = {
    'epochs': 2,
    'batch_size': 2
}

SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
TRAIN_FOLDER = '../nesmdb24_exprsco/train'
VAL_FOLDER = '../nesmdb24_exprsco/valid'
TEST_FOLDER = '../nesmdb24_exprsco/test'

TRAIN_FOLDER = os.path.join(SCRIPT_FOLDER, TRAIN_FOLDER)
VAL_FOLDER = os.path.join(SCRIPT_FOLDER, VAL_FOLDER)
TEST_FOLDER = os.path.join(SCRIPT_FOLDER, TEST_FOLDER)

def tee(*args, files=[sys.stdout], **kwargs):
    for f in files:
        print(*args, file=f, **kwargs)

class ExprCriterion:
    def __init__(self):
        self.sos_eos_criterion = nn.CrossEntropyLoss(ignore_index=3)
        self.note_criterions = [nn.CrossEntropyLoss(ignore_index=dim)
                for dim in DIMENSIONS]
        self.num_notes = float(len(self.note_criterions))

    def __call__(self, logits, note_targets, sos_eos_targets):
        loss = self.sos_eos_criterion(logits[:, -3:], sos_eos_targets.view(-1))

        offset = 0
        for dim, criterion, targets in zip(DIMENSIONS, self.note_criterions, note_targets):
            loss += criterion(logits[:, offset:offset + dim], targets.view(-1)) / self.num_notes
            offset += dim

        return loss

def train_epoch(model, optimizer, train_loader, val_loader, cfg):
    criterion = cfg['criterion']

    for minibatch_count, (input, note_targets, sos_eos_targets, lengths) in \
            enumerate(train_loader):
        model.train()
        input = input.to(cfg['computing_device'])
        for i in range(len(note_targets)):
            note_targets[i] = note_targets[i].to(cfg['computing_device'])
        sos_eos_targets = sos_eos_targets.to(cfg['computing_device'])
        optimizer.zero_grad()

        output_logits = model(input, lengths)
        loss = criterion(output_logits, note_targets, sos_eos_targets)

        tee('Training loss:', loss.item())

        loss.backward()
        optimizer.step()


def validation_pass(val_loader, cfg):
    pass

def train(cfg):
    train_data = NESExprDataset(TRAIN_FOLDER)
    train_loader = DataLoader(train_data, batch_size=cfg['batch_size'],
            sampler=SubsetRandomSampler(np.arange(len(train_data))),
            collate_fn=collate_expressive)

    val_data = NESExprDataset(VAL_FOLDER)
    val_loader = DataLoader(val_data, batch_size=cfg['batch_size'],
            sampler=SubsetRandomSampler(np.arange(len(val_data))),
            collate_fn=collate_expressive)

    cfg['criterion'] = ExprCriterion()
    
    model = BaselineLSTM()
    optimizer = optim.Adam(model.parameters())
    for i in range(cfg['epochs']):
        train_epoch(model, optimizer, train_loader, val_loader, cfg)


def add_all_arguments(parser):
    parser.add_argument('--action', '-a', default='train',
            choices=['train'])
    parser.add_argument('--config-file', '-c', default=None)


def load_config(filename):
    raise NotImplementedError('Haven\'t implemented non-default config')

def main():
    parser = argparse.ArgumentParser()
    add_all_arguments(parser)
    args = parser.parse_args()

        
    if args.config_file is None:
        cfg = DEFAULT_CONFIG
    else:
        cfg = load_config(args.config_file)

    if torch.cuda.is_available():
        cfg['computing_device'] = torch.device('cuda')
        tee('Using CUDA')
    else:
        cfg['computing_device'] = torch.device('cpu')
        tee('Not using CUDA')

    if args.action == 'train':
        train(cfg)

if __name__ == '__main__':
    main()
