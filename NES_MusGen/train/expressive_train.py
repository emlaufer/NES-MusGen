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
import itertools

# Options that can be changed between different runs of the model
DEFAULT_RUN_CONFIG = {
    'epochs': 5,
    'batch_size': 8,
    'validation_freq': 30,
    'validation_prop': 0.25
}

# Options that can't be changed between different runs of the model
DEFAULT_MODEL_CONFIG = {
    'hidden_size': 1024
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

def train_epoch(model, optimizer, train_loader, val_loader, state, cfg,
        starting_minibatch=0):
    criterion = cfg['criterion']

    losses = None

    # TODO: we want to be able to resume training mid epoch. Is this the right
    # way to do it?
    state['random_state'] = torch.get_rng_state()

    for minibatch_count, (input, note_targets, sos_eos_targets, lengths) in \
            itertools.islice(enumerate(train_loader), starting_minibatch, None):

        state['minibatch_count'] = minibatch_count
        if minibatch_count % cfg['validation_freq'] == 0:
            validate_and_checkpoint(model, optimizer, val_loader, cfg, state)

            losses = []
            state['training_losses'].append({
                'epoch': state['epoch'],
                'minibatch': state['minibatch_count'],
                'losses': losses
                })
            
        model.train()
        input = input.to(cfg['computing_device'])
        for i in range(len(note_targets)):
            note_targets[i] = note_targets[i].to(cfg['computing_device'])
        sos_eos_targets = sos_eos_targets.to(cfg['computing_device'])
        optimizer.zero_grad()

        output_logits = model(input, lengths)
        loss = criterion(output_logits, note_targets, sos_eos_targets)

        tee('Minibatch', minibatch_count, ' Training loss:', loss.item(),
                files=cfg['out_files'])

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        for f in cfg['out_files']: f.flush()


    validate_and_checkpoint(model, optimizer, val_loader, cfg, state)

def validate_and_checkpoint(model, optimizer, val_loader, cfg, state):
    avg_val_loss = validation_pass(model, val_loader, cfg, state)

    state['model_state_dict'] = model.state_dict().copy()
    state['optim_state_dict'] = optimizer.state_dict().copy()

    if avg_val_loss < state['min_validation_loss']:
        state['min_validation_loss'] = avg_val_loss
        state['best_minibatch'] = state['minibatch_count']
        state['best_epoch'] = state['epoch']

        if cfg['best_model_filename'] is not None:
            torch.save(state, cfg['best_model_filename'])

    if cfg['latest_model_filename'] is not None:
        torch.save(state, cfg['latest_model_filename'])

    tee('Average validation loss:', avg_val_loss, files=cfg['out_files'])
    tee(files=cfg['out_files'])



def validation_pass(model, val_loader, cfg, state):
    criterion = cfg['criterion']
    val_prop = cfg['validation_prop']

    model.eval()
    with torch.no_grad():
        losses = []
        for val_count, (input, note_targets, sos_eos_targets, lengths) \
                in enumerate(val_loader):

            if val_count >= val_prop * len(val_loader): break

            input = input.to(cfg['computing_device'])
            for i in range(len(note_targets)):
                note_targets[i] = note_targets[i].to(cfg['computing_device'])
            sos_eos_targets = sos_eos_targets.to(cfg['computing_device'])

            output_logits = model(input, lengths)
            loss = criterion(output_logits, note_targets, sos_eos_targets)

            tee('Minibatch', val_count, ' Validation loss', loss.item(),
                    files=cfg['out_files'])

            losses.append(loss.item())

            for f in cfg['out_files']: f.flush()

        state['validation_losses'].append({
                'epoch': state['epoch'],
                'minibatch': state['minibatch_count'],
                'losses': losses
                })

    model.train()

    return np.mean(losses)


def train(model, state, cfg):
    train_data = NESExprDataset(TRAIN_FOLDER)
    train_loader = DataLoader(train_data, batch_size=cfg['batch_size'],
            sampler=SubsetRandomSampler(np.arange(len(train_data))),
            collate_fn=collate_expressive)

    val_data = NESExprDataset(VAL_FOLDER)
    val_loader = DataLoader(val_data, batch_size=cfg['batch_size'],
            sampler=SubsetRandomSampler(np.arange(len(val_data))),
            collate_fn=collate_expressive)

    cfg['criterion'] = ExprCriterion()
    
    optimizer = optim.Adam(model.parameters())
    if cfg['optim_state_dict'] is not None:
        optimizer.load_state_dict(cfg['optim_state_dict'])

    for i in range(cfg['epochs']):
        state['epoch'] = i
        train_epoch(model, optimizer, train_loader, val_loader, state, cfg)


def add_all_arguments(parser):
    parser.add_argument('--action', '-a', default='train',
            choices=['train'])
    parser.add_argument('--config-file', '-c', default=None)
    parser.add_argument('--model-infile', '-m', default=None)
    parser.add_argument('--best-model', '-b', default='__best_model.pt')
    parser.add_argument('--latest-model', '-l', default='__latest_model.pt')
    parser.add_argument('--prog-outfile', '-p', action='append', default=[])


def load_config(filename):
    raise NotImplementedError('Haven\'t implemented non-default config')

def main():
    parser = argparse.ArgumentParser()
    add_all_arguments(parser)
    args = parser.parse_args()

    out_files = [sys.stdout]
    out_files.extend([open(filename, 'w') for filename in args.prog_outfile])

    try:
        if args.config_file is None:
            run_config, model_config = (DEFAULT_RUN_CONFIG, DEFAULT_MODEL_CONFIG)
        else:
            run_config, model_config = load_config(args.config_file)

        cfg = model_config.copy()
        cfg.update(run_config)

        cfg['best_model_filename'] = args.best_model
        cfg['latest_model_filename'] = args.latest_model
        cfg['out_files'] = out_files

        if torch.cuda.is_available():
            cfg['computing_device'] = torch.device('cuda')
            tee('Using CUDA', files=out_files)
        else:
            cfg['computing_device'] = torch.device('cpu')
            tee('Not using CUDA', files=out_files)

        state_model_config = model_config.copy()
        state = {'min_validation_loss': 10000.,
                'training_losses': [],
                'validation_losses': [],
                'run_config': run_config,
                'model_config': state_model_config,
                'epoch': 1
                }

        model = BaselineLSTM(hidden_dim=cfg['hidden_size'])
        cfg['optim_state_dict'] = None
        if args.model_infile is not None:
            state = torch.load(args.model_infile,
                    map_location=lambda storage, loc: storage)
            state['run_config'] = run_config

            for key in state['model_config']:
                if state_model_config[key] != state['config'][key]:
                    raise ValueError('New config does not match checkpoint config')

            state['model_config'] = state_model_config

            model.load_state_dict(state['model_state_dict'])
            cfg['optim_state_dict'] = state['optim_state_dict']

        state['random_seed'] = torch.initial_seed()
        
        model = model.to(cfg['computing_device'])

        if args.action == 'train':
            train(model, state, cfg)
    finally:
        for f in out_files[1:]:
            f.close()

if __name__ == '__main__':
    main()
