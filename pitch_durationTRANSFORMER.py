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
"""

# library for understanding music
from music21 import *
import torch
import torch.nn as nn
import numpy as np
import math
from dataset_funct import ImprovDurationDataset, ImprovPitchDataset, convert_to_midi, read_midi_pitch, read_midi_duration 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

# TRANSFORMER MODEL
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
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

import time
def train(model, vocab, train_data, criterion, optimizer):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(vocab)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
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
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source, vocab):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(vocab)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


if __name__ == '__main__':

    #%% DATA LOADING
    
    # LOAD PITCH DATASET
    datasetPitch = ImprovPitchDataset()
    X_pitch = datasetPitch.getData()
    # set vocabulary for conversion
    vocabPitch = datasetPitch.vocab
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
    datasetDuration = ImprovDurationDataset()
    X_duration = datasetDuration.getData()
    # set vocabulary for conversion
    vocabDuration = datasetDuration.vocab
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
        
        """
        new_data = []
        for sequence in data:
            for note in sequence:
                new_data.append(note)
        new_data = np.array(new_data)
        
        
        data = torch.tensor([dict_to_ix[w] for w in new_data], dtype=torch.long)
        """
        
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
    bptt = 35 # lenght of a sequence of data (IMPROVEMENT HERE!!)
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len] # input 
        target = source[i+1:i+1+seq_len].view(-1) # target (same as input but shifted by 1)
        return data, target
    
    #%% PITCH MODEL TRAINING
    
    # HYPERPARAMETERS
    ntokens_pitch = len(vocabPitch) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    modelPitch = TransformerModel(ntokens_pitch, emsize, nhead, nhid, nlayers, dropout).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 10 # The number of epochs
    best_model = None
    
    # TRAINING LOOP
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(modelPitch, vocabPitch, train_data_pitch, criterion, optimizer)
        val_loss = evaluate(modelPitch, val_data_pitch, vocabPitch)
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
    test_loss = evaluate(best_model_pitch, test_data_pitch, vocabPitch)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    

    #%% DURATION MODEL TRAINING
    
    # HYPERPARAMETERS
    ntokens_duration = len(vocabDuration) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    modelDuration = TransformerModel(ntokens_duration, emsize, nhead, nhid, nlayers, dropout).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(modelDuration.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 10 # The number of epochs
    best_model = None
    
    # TRAINING LOOP
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(modelDuration, vocabDuration, train_data_duration, criterion, optimizer)
        val_loss = evaluate(modelDuration, val_data_duration, vocabDuration)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_duration = modelDuration
    
        scheduler.step()
    
    # TEST THE MODEL
    test_loss = evaluate(best_model_duration, test_data_duration, vocabDuration)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    

    #%% SAMPLES GENERATION

    
    def getNote(val, dict_to_ix): 
        for key, value in dict_to_ix.items(): 
             if val == value: 
                 return key

    def generate(model, melody4gen, dict_to_ix, next_notes=10):
        melody4gen = melody4gen.tolist()
        for i in range(0,next_notes):
            x_pred = torch.tensor([dict_to_ix[w] for w in melody4gen], dtype=torch.long)
            y_pred = model(x_pred)
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            melody4gen.append(getNote(word_index, dict_to_ix))
        return melody4gen
    
    
    #notes2gen = 40 # number of new notes to generate
    #row_of_test = 0
    #melody4gen_pitch = test_pitch[row_of_test][:80] # pitch of the input for generation
    #melody4gen_duration = test_duration[row_of_test][:80] # duration of the input for generation
    #new_melody_pitch = generate(modelPitch, melody4gen_pitch, pitch_to_ix, next_notes=notes2gen)
    #new_melody_duration = generate(modelDuration, melody4gen_duration, duration_to_ix, notes2gen)


    #print(melody4gen_pitch)
    
    #convert_to_midi(new_melody_pitch, new_melody_duration)

    
    # Convert the output of the model to midi
    def convert_to_midi(new_melody_pitch, new_melody_duration):
        offset = 0
        output_notes = []
        #output_notes = stream.Stream()
        # create note and chord objects based on the values generated by the model
        for i in range(len(new_melody_pitch)):
            if new_melody_pitch[i] == 'R':
                new_note = note.Rest()
                new_note.offset = offset
                #new_note.duration.quarterLength = float(new_melody_duration[i])
            else: 
                new_note = note.Note()
                new_note.pitch.nameWithOctave = new_melody_pitch[i]
                new_note.offset = offset
                #new_note.duration.quarterLength = float(new_melody_duration[i])
                new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            offset += float(new_melody_duration[i])
            
        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='output/music.mid')  

    
    #specify the path
    f='data/w_jazz/ArtPepper_BluesForBlanche_FINAL.mid'
    melody4gen_pitch = (read_midi_pitch(f))[:80]
    melody4gen_duration = (read_midi_duration(f))[:80]
    print(melody4gen_pitch)
    print(melody4gen_duration)
    
    notes2gen = 40 # number of new notes to generate
    new_melody_pitch = generate(modelPitch, melody4gen_pitch, pitch_to_ix, next_notes=notes2gen)
    new_melody_duration = generate(modelDuration, melody4gen_duration, duration_to_ix, notes2gen)

    convert_to_midi(new_melody_pitch, new_melody_duration)

    #convert_to_midi(melody4gen_pitch, melody4gen_duration)
    
    
    # SAVE MODELS
    
    savePATH = 'modelsPitch/modelPitch.pt'
    torch.save(modelPitch, savePATH)
    savePATH = 'modelsDuration/modelDuration.pt'
    torch.save(modelDuration, savePATH)
    
    
    #%% MODEL EVALUATION
    # accuracy, perplexity (paper seq-Attn)
    # BLEU, MGeval (paper explicitly conditioned melody generation)
    
    
    
    