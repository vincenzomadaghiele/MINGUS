#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:56:44 2020

@author: vincenzomadaghiele
"""
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
import time
import pretty_midi

import MINGUS_dataset_funct as dt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


# TRANSFORMER MODEL
class Transformer(nn.Module):

    def __init__(self, 
                 ntoken, 
                 ninp, 
                 nhead, 
                 n_enc, 
                 n_dec, 
                 forward_expansion, 
                 src_pad_idx, 
                 device, 
                 dropout=0.5):
        super(Transformer, self).__init__()
        #from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.ninp = ninp
        
        self.src_embedding = nn.Embedding(ntoken, ninp)
        self.src_pos_encoder = PositionalEncoding(ninp, dropout)
        self.trg_embedding = nn.Embedding(ntoken, ninp)
        self.trg_pos_encoder = PositionalEncoding(ninp, dropout)
        
        self.transformer = nn.Transformer(
            ninp,
            nhead,
            n_enc,
            n_dec,
            forward_expansion,
            dropout,
        )
        
        self.fc_out = nn.Linear(ninp, ntoken)
        #self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_pad_mask(self, src):
        pad_mask = src.transpose(0, 1) == self.src_pad_idx
        #pad_mask = pad_mask.float().masked_fill(pad_mask == True, float('-inf')).masked_fill(pad_mask == False, float(0.0))
        return pad_mask.to(self.device)

    def init_weights(self):
        initrange = 0.1
        self.transformer.weight.data.uniform_(-initrange, initrange)
        self.transformer.bias.data.zero_()
        self.transformer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg, trg_mask):
        src_padding_mask = self.make_pad_mask(src)
        tgt_padding_mask = self.make_pad_mask(trg)
        
        src = self.src_embedding(src) * math.sqrt(self.ninp)
        trg = self.trg_embedding(trg) * math.sqrt(self.ninp)
        src = self.src_pos_encoder(src)
        trg = self.trg_pos_encoder(trg)
        
        output = self.transformer(src, 
                                  trg, 
                                  tgt_mask=trg_mask, 
                                  src_key_padding_mask=src_padding_mask, 
                                  tgt_key_padding_mask=tgt_padding_mask)

        output = self.fc_out(output)
        return output

# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

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
    target = source[i+1:i+1+seq_len]#.view(-1) # target (same as input but shifted by 1)

    return data, target

def train(model, vocab, train_data, criterion, optimizer, bptt, device, epoch, scheduler):
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
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        optimizer.zero_grad()
        
        if targets.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(targets.size(0)).to(device)
        
        output = model(data, targets, src_mask) 
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
        
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
                    epoch, batch, len(train_data) // bptt, 
                    #scheduler.get_last_lr()[0],
                    3.54,
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source, vocab, bptt, device, criterion):
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
    src_mask = eval_model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            if data.size(0) != bptt:
                src_mask = eval_model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, targets, src_mask) 
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets.view(-1)).item()
    return total_loss / (len(data_source) - 1)


#%%

if __name__ == '__main__':
    # DATA LOADING
    
    # LOAD PITCH DATASET
    pitch_path = 'data/w_jazz/'
    datasetPitch = dt.ImprovPitchDataset(pitch_path, 20)
    X_pitch = datasetPitch.getData()
    # set vocabulary for conversion
    vocabPitch = datasetPitch.vocab
    # Add padding tokens to vocab
    vocabPitch.append('<pad>')
    #vocabPitch.append('<sos>')
    #vocabPitch.append('<eos>')
    pitch_to_ix = {word: i for i, word in enumerate(vocabPitch)}
    #print(X_pitch[:3])
    
    # Divide pitch into train, validation and test
    train_pitch = X_pitch[:int(len(X_pitch)*0.7)]
    val_pitch = X_pitch[int(len(X_pitch)*0.7)+1:int(len(X_pitch)*0.7)+1+int(len(X_pitch)*0.1)]
    test_pitch = X_pitch[int(len(X_pitch)*0.7)+1+int(len(X_pitch)*0.1):]
    
    # LOAD DURATION DATASET
    duration_path = 'data/w_jazz/'
    datasetDuration = dt.ImprovDurationDataset(duration_path, 10)
    X_duration = datasetDuration.getData()
    # set vocabulary for conversion
    vocabDuration = datasetDuration.vocab
    # Add padding tokens to vocab
    vocabDuration.append('<pad>')
    #vocabDuration.append('<sos>')
    #vocabDuration.append('<eos>')
    duration_to_ix = {word: i for i, word in enumerate(vocabDuration)}
    #print(X_duration[:3])
    
    # Divide duration into train, validation and test
    train_duration = X_duration[:int(len(X_duration)*0.7)]
    val_duration = X_duration[int(len(X_duration)*0.7)+1:int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1)]
    test_duration = X_duration[int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1):]
    
    #%% DATA PREPARATION
    
    # Melody segmentation
    segment_length = 35 # Maximum value of a sequence
    train_pitch_segmented, train_duration_segmented = dt.segmentDataset(train_pitch, train_duration, segment_length)
    val_pitch_segmented, val_duration_segmented = dt.segmentDataset(val_pitch, val_duration, segment_length)
    test_pitch_segmented, test_duration_segmented = dt.segmentDataset(test_pitch, test_duration, segment_length)

    
    batch_size = 20
    eval_batch_size = 10
    
    # pad, batch and convert to tensor
    train_data_pitch = dt.batchify(train_pitch_segmented, batch_size, pitch_to_ix, device)
    val_data_pitch = dt.batchify(val_pitch_segmented, eval_batch_size, pitch_to_ix, device)
    test_data_pitch = dt.batchify(test_pitch_segmented, eval_batch_size, pitch_to_ix, device)
    
    train_data_duration = dt.batchify(train_duration_segmented, batch_size, duration_to_ix, device)
    val_data_duration = dt.batchify(val_duration_segmented, eval_batch_size, duration_to_ix, device)
    test_data_duration = dt.batchify(test_duration_segmented, eval_batch_size, duration_to_ix, device)
    
    # a padded sequence is of length segment_value + 2 (sos and eos tokens)
    bptt = segment_length #+ 2 # lenght of a sequence of data (IMPROVEMENT HERE!!)


    #%% PITCH MODEL
    
    ntoken = len(vocabPitch)
    ninp = 200 # embedding dimension
    nhead = 4 # number of attention heads
    n_enc = 6 # number of encoder layers
    n_dec = 6 # number of decoder layers
    forward_expansion = 4 
    pad_idx = pitch_to_ix['<pad>']
    device = device
    dropout=0.5
    
    modelPitch = Transformer(ntoken, ninp, nhead, n_enc, n_dec, forward_expansion,
                            pad_idx, device, dropout).to(device)
    
    # Training hyperparameters
    num_epochs = 10
    learning_rate = 3e-4
    batch_size = 32
    best_val_loss = float("inf")
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    #lr = 5.0 # learning rate
    #optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
     
    optimizer = optim.Adam(modelPitch.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )
    
    # TRAINING LOOP
    for epoch in range(1, num_epochs + 1):
        
        epoch_start_time = time.time()
        train(modelPitch, vocabPitch, train_data_pitch, criterion, optimizer, 
                  bptt, device, epoch, scheduler)
        val_loss = evaluate(modelPitch, val_data_pitch, vocabPitch, 
                                bptt, device, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_pitch = modelPitch
    
        scheduler.step()
    
    
    # TEST THE MODEL
    test_loss = evaluate(best_model_pitch, test_data_pitch, vocabPitch,  
                             bptt, device, criterion)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    
    savePATHpitch = 'modelsPitch/transformerPitch_'+ str(num_epochs) + 'epochs_wjazz_segmented.pt'
    state_dictPitch = best_model_pitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    
    
    #%% DURATION MODEL
    
    ntoken = len(vocabDuration)
    ninp = 200 # embedding dimension
    nhead = 4 # number of attention heads
    n_enc = 6 # number of encoder layers
    n_dec = 6 # number of decoder layers
    forward_expansion = 4 
    pad_idx = pitch_to_ix['<pad>']
    device = device
    dropout=0.5
    
    modelDuration = Transformer(ntoken, ninp, nhead, n_enc, n_dec, forward_expansion,
                            pad_idx, device, dropout).to(device)
    
    # Training hyperparameters
    num_epochs = 10
    learning_rate = 3e-4
    batch_size = 32
    best_val_loss = float("inf")
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    #lr = 5.0 # learning rate
    #optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    optimizer = optim.Adam(modelDuration.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )
    
    # TRAINING LOOP
    for epoch in range(1, num_epochs + 1):
        
        epoch_start_time = time.time()
        train(modelDuration, vocabDuration, train_data_duration, criterion, optimizer, 
                  bptt, device, epoch, scheduler)
        val_loss = evaluate(modelDuration, val_data_duration, vocabDuration, 
                                bptt, device, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_pitch = modelDuration
    
        scheduler.step()
    
    
    # TEST THE MODEL
    test_loss = evaluate(modelDuration, val_data_duration, vocabDuration,  
                             bptt, device, criterion)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    
    savePATHpitch = 'modelsPitch/transformerDuration_'+ str(num_epochs) + 'epochs_wjazz_segmented.pt'
    state_dictPitch = best_model_pitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
        
    