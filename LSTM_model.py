#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:00:45 2021

@author: vincenzomadaghiele
"""

import os.path as op
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.autograd import Variable
import time
import math

#sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
#import utils.constants as const

torch.manual_seed(1)

class NoCondLSTM(nn.Module):
    def __init__(self, vocab_size=None, embed_dim=None, output_dim=None, hidden_dim=None, seq_len=None, 
            num_layers=None, batch_size=None, dropout=0.5, batch_norm=True, no_cuda=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.no_cuda = no_cuda

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=self.num_layers, 
                            batch_first=False, dropout=dropout)
        mid_dim = (hidden_dim + output_dim) // 2
        self.decode1 = nn.Linear(hidden_dim, mid_dim)
        self.decode_bn = nn.BatchNorm1d(seq_len)
        self.decode2 = nn.Linear(mid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)

        self.hidden_and_cell = None
        if batch_size is not None:
            self.init_hidden_and_cell(batch_size)

        if torch.cuda.is_available() and (not self.no_cuda):
            self.cuda()

    def init_hidden_and_cell(self, batch_size):
        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        cell = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        if torch.cuda.is_available() and (not self.no_cuda):
            hidden = hidden.cuda()
            cell = cell.cuda()
        self.hidden_and_cell = (hidden, cell)

    def repackage_hidden_and_cell(self):
        new_hidden = Variable(self.hidden_and_cell[0].data)
        new_cell = Variable(self.hidden_and_cell[1].data)
        if torch.cuda.is_available() and (not self.no_cuda):
            new_hidden = new_hidden.cuda()
            new_cell = new_cell.cuda()
        self.hidden_and_cell = (new_hidden, new_cell)

    def forward(self, data, *args, **kwargs):
        embedded = self.embedding(data)
        lstm_out, self.hidden_and_cell = self.lstm(embedded, self.hidden_and_cell)
        decoded = self.decode1(lstm_out)
        if self.batch_norm:
            decoded = self.decode_bn(decoded)
        decoded = self.decode2(F.relu(decoded))
        output = self.softmax(decoded)
        return output



def train(model, vocab, train_data, criterion, optimizer, scheduler, epoch, bptt):
    '''

    Parameters
    ----------
    model : pytorch Model
        Model to be trained.
    vocab : python dictionary
        dictionary of the tokens used in training.
    train_data : pytorch Tensor
        batched data to be used in training.
    criterion : pytorch Criterion
        criterion to be used for otimization.
    optimizer : pytorch Optimizer
        optimizer used in training.

    Returns
    -------
    None.

    '''
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(vocab)
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets, _ = get_batch(train_data, i, bptt)
        #print(data.shape)
        #print('bptt ', bptt)
        # there was an issue with batch size
        # change data.shape[0] to seq_len instead of batch size
        #if data.shape[0] == bptt and data.shape[1] == bptt:
        #print('inside')
        model.init_hidden_and_cell(data.shape[1]) # initialize hidden to the dimension of the batch
        optimizer.zero_grad()
        output = model(data) 
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    
        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                      elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source, vocab, criterion, bptt, device):
    '''

    Parameters
    ----------
    eval_model : pytorch Model
        Model, already trained to be evaluated.
    data_source : pytorch Tensor
        Evaluation dataset.
    vocab : python dictionary
        dictionary used for tokenization.

    Returns
    -------
    float
        total loss of the model on validation/test data.

    '''
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(vocab)
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets, _ = get_batch(data_source, i, bptt)
            #if data.shape[0] == bptt and data.shape[1] == bptt:
            eval_model.init_hidden_and_cell(data.shape[1])
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def get_batch(source, i, bptt):
    '''

    Parameters
    ----------
    source : pytorch Tensor
        source of batched data.
    i : integer
        number of the batch to be selected.

    Returns
    -------
    data : pytorch Tensor
        batch of data of size [bptt x bsz].
    target : pytorch Tensor
        same as data but sequences are shifted by one token, 
        of size [bptt x bsz].

    '''
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len] # input 
    target = source[i+1:i+1+seq_len].view(-1) # target (same as input but shifted by 1)
    targets_no_reshape = source[i+1:i+1+seq_len]

    return data, target, targets_no_reshape