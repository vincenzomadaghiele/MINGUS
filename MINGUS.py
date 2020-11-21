#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:47:05 2020

@author: vincenzomadaghiele

This scripts recreates the 'no_cond' model of the paper:
    Explicitly Conditioned Melody Generation: A Case Study with Interdependent RNNs
    (Genchel et al., 2019)
but using Transformers instead of RNNs
and with a different dataset

This code is used for training and generation of samples

Things to do:
    - tell the model to identify '<pad>' tokens!
    - implement remaining metrics (BLEU, MGEval)
    - run model with other datasets (folkDB, seAttn data) and compare metrics
    - make net scheme
    - grid search for model optimization
    - conditioning on chords and inter-conditioning between pitch and duration
    - move all constants to an external .py file
    
For next meeting:
    - it is NOT USEFUL to train the duration model on the augmented data, 
        might as well train it on the not augumented data!
    - problem: my 35-tokens long sequences do not have a <sos> and <eos> 
        token at their start as with translation models 
        (in their case the sentences are not divided into pieces as in my case)
    
Next training:
    - include pad tokens as in translation tutorial
    - make system to name different trained net
    
"""

import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim
#from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
#from torch.utils.tensorboard import SummaryWriter

import numpy as np
import math
from MINGUS_dataset_funct import ImprovDurationDataset, ImprovPitchDataset, readMIDI, convertMIDI

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

# TRANSFORMER MODEL
class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


import time
def train(model, vocab, train_data, criterion, optimizer):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(vocab)
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
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
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source, vocab):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(vocab)
    src_mask = eval_model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = eval_model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


if __name__ == '__main__':

    # DATA LOADING
    
    # LOAD PITCH DATASET
    dataset_path = 'data/w_jazz/'
    datasetPitch = ImprovPitchDataset(dataset_path, 20)
    X_pitch = datasetPitch.getData()
    # set vocabulary for conversion
    vocabPitch = datasetPitch.vocab
    # Add padding tokens to vocab
    vocabPitch.append('<pad>')
    vocabPitch.append('<sos>')
    vocabPitch.append('<eos>')
    pitch_to_ix = {word: i for i, word in enumerate(vocabPitch)}
    #print(X_pitch[:3])
    
    # Divide pitch into train, validation and test
    train_pitch = X_pitch[:int(len(X_pitch)*0.7)]
    val_pitch = X_pitch[int(len(X_pitch)*0.7)+1:int(len(X_pitch)*0.7)+1+int(len(X_pitch)*0.1)]
    test_pitch = X_pitch[int(len(X_pitch)*0.7)+1+int(len(X_pitch)*0.1):]
    
    # LOAD DURATION DATASET
    datasetDuration = ImprovDurationDataset(dataset_path, 10)
    X_duration = datasetDuration.getData()
    # set vocabulary for conversion
    vocabDuration = datasetDuration.vocab
    # Add padding tokens to vocab
    vocabDuration.append('<pad>')
    vocabDuration.append('<sos>')
    vocabDuration.append('<eos>')
    duration_to_ix = {word: i for i, word in enumerate(vocabDuration)}
    #print(X_duration[:3])
    
    # Divide duration into train, validation and test
    train_duration = X_duration[:int(len(X_duration)*0.7)]
    val_duration = X_duration[int(len(X_duration)*0.7)+1:int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1)]
    test_duration = X_duration[int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1):]
    
    
    
    #%% DATA PREPARATION

    # pad data to max_lenght of sequences, prepend <sos> and append <eos>
    def pad(data):
        # from: https://pytorch.org/text/_modules/torchtext/data/field.html
        data = list(data)
        # calculate max lenght
        max_len = max(len(x) for x in data)
        # Define padding tokens
        pad_token = '<pad>'
        init_token = '<sos>'
        eos_token = '<eos>'
        # pad each sequence in the data to max_lenght
        padded, lengths = [], []
        for x in data:
            padded.append(
                ([init_token])
                + list(x[:max_len])
                + ([eos_token])
                + [pad_token] * max(0, max_len - len(x)))
        lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        return padded
    
    # divide into batches of size bsz and converts notes into numbers
    def batchify(data, bsz, dict_to_ix):
        
        padded = pad(data)
        padded_num = [[dict_to_ix[x] for x in ex] for ex in padded]
        
        data = torch.tensor(padded_num, dtype=torch.long)
        data = data.contiguous()
        
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)
    
    batch_size = 20
    eval_batch_size = 10
    
    train_data_pitch = batchify(train_pitch, batch_size, pitch_to_ix)
    val_data_pitch = batchify(val_pitch, eval_batch_size, pitch_to_ix)
    test_data_pitch = batchify(test_pitch, eval_batch_size, pitch_to_ix)
    
    train_data_duration = batchify(train_duration, batch_size, duration_to_ix)
    val_data_duration = batchify(val_duration, eval_batch_size, duration_to_ix)
    test_data_duration = batchify(test_duration, eval_batch_size, duration_to_ix)
    
    # divide into target and input sequence of lenght bptt
    # --> obtain matrices of size bptt x batch_size 
    bptt = 35 
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len] # input 
        target = source[i+1:i+1+seq_len] #.view(-1) # target (same as input but shifted by 1)
        return data, target
    
    
    #%% PITCH MODEL TRAINING
    
    # Training hyperparameters
    num_epochs = 10
    learning_rate = 5.0
    
    # Model hyperparameters
    src_vocab_size = len(vocabPitch)
    trg_vocab_size = len(vocabPitch)
    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    max_len = 100
    forward_expansion = 4
    src_pad_idx = pitch_to_ix['<pad>']
    
    modelPitch = Transformer(embedding_size, src_vocab_size, trg_vocab_size, 
                        src_pad_idx, num_heads, num_encoder_layers,
                        num_decoder_layers, forward_expansion, dropout, 
                        max_len, device).to(device)
    
    optimizer = optim.Adam(modelPitch.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True)
    
    pad_idx = pitch_to_ix['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Tensorboard to get nice loss plot
    #writer = SummaryWriter("runs/loss_plot")
    step = 0
    start_time = time.time()
    total_loss = 0.
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")
        
        modelPitch.train()
        losses = []
    
        for batch, i in enumerate(range(0, train_data_pitch.size(0) - 1, bptt)):
            data, targets = get_batch(train_data_pitch, i)
            
            # Get input and targets and get to cuda
            #inp_data = batch.src.to(device)
            #target = batch.trg.to(device)
    
            # Forward prop
            output = modelPitch(data, targets)
    
            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            targets = targets.reshape(-1)
    
            optimizer.zero_grad()
    
            loss = criterion(output, targets)
            losses.append(loss.item())
    
            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(modelPitch.parameters(), max_norm=1)
    
            # Gradient descent step
            optimizer.step()
            
            # Print training information
            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      ' ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data_pitch) // bptt,
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
            
            # plot to tensorboard
            #writer.add_scalar("Training loss", loss, global_step=step)
            step += 1
    
        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)
        
    
    savePATHpitch = 'modelsPitch/MINGUSpitch_'+ str(num_epochs) + 'epochs.pt'
    state_dictPitch = modelPitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    

    #%% DURATION MODEL TRAINING
    
    # Training hyperparameters
    num_epochs = 10
    learning_rate = 5.0
    
    # Model hyperparameters
    src_vocab_size = len(vocabDuration)
    trg_vocab_size = len(vocabDuration)
    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    max_len = 100
    forward_expansion = 4
    src_pad_idx = duration_to_ix['<pad>']
    
    modelDuration = Transformer(embedding_size, src_vocab_size, trg_vocab_size, 
                        src_pad_idx, num_heads, num_encoder_layers,
                        num_decoder_layers, forward_expansion, dropout, 
                        max_len, device).to(device)
    
    
    optimizer = optim.Adam(modelDuration.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True)
    
    pad_idx = duration_to_ix['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Tensorboard to get nice loss plot
    #writer = SummaryWriter("runs/loss_plot")
    step = 0
    start_time = time.time()
    total_loss = 0.
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")
        
        modelDuration.train()
        losses = []
    
        for batch, i in enumerate(range(0, train_data_duration.size(0) - 1, bptt)):
            data, targets = get_batch(train_data_duration, i)
            
            # Get input and targets and get to cuda
            #inp_data = batch.src.to(device)
            #target = batch.trg.to(device)
    
            # Forward prop
            output = modelDuration(data, targets)
    
            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            targets = targets.reshape(-1)
    
            optimizer.zero_grad()
    
            loss = criterion(output, targets)
            losses.append(loss.item())
    
            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(modelDuration.parameters(), max_norm=1)
    
            # Gradient descent step
            optimizer.step()
            
            # Print training information
            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  ' ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data_duration) // bptt, 
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
    
            # plot to tensorboard
            #writer.add_scalar("Training loss", loss, global_step=step)
            step += 1
    
        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)
    
    savePATHduration = 'modelsDuration/MINGUSduration_'+ str(num_epochs) + 'epochs.pt'
    state_dictDuration = modelDuration.state_dict()
    torch.save(state_dictDuration, savePATHduration)
    

    