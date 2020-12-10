#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:41:11 2020

@author: vincenzomadaghiele

from: pytorch embedding tutorial
"""

# library for understanding music
from music21 import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

# define the Dataset object for Pytorch
class ImprovDataset(Dataset):
    
    """
    --> DataLoader can do the batch computation for us

    Implement a custom Dataset:
    inherit Dataset
    implement __init__ , __getitem__ , and __len__
    """
    
    def __init__(self):
        #for listing down the file names
        import os
        
        #specify the path
        path='data/w_jazz/'
        #read all the filenames
        files=[i for i in os.listdir(path) if i.endswith(".mid")]
        #reading each midi file
        notes_array = np.array([read_midi_duration(path+i) for i in files])
        
        #converting 2D array into 1D array
        notes_ = [element for note_ in notes_array for element in note_]
        #No. of unique notes
        unique_notes = list(set(notes_))
        print("number of unique notes: " + str(len(unique_notes)))
        
        from collections import Counter
        #computing frequency of each note
        freq = dict(Counter(notes_))
              
        # the threshold for frequent notes can change 
        threshold=50 # this threshold is the number of classes that have to be predicted
        frequent_notes = [note_ for note_, count in freq.items() if count>=threshold]
        print("number of frequent notes (more than 50 times): " + str(len(frequent_notes)))
        self.num_frequent_notes = len(frequent_notes)
        self.vocab = frequent_notes
        
        # prepare new musical files which contain only the top frequent notes
        new_music=[]
        for notes in notes_array:
            temp=[]
            for note_ in notes:
                if note_ in frequent_notes:
                    temp.append(note_)            
            new_music.append(temp)
        new_music = np.array(new_music) # same solos but with only most frequent notes

        self.x = new_music

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def getData(self):
        return self.x

#defining function to read MIDI files
def read_midi_duration(file):
    
    print("Loading Music File:",file)
    
    notes=[]
    notes_to_parse = None
    
    #parsing a midi file
    midi = converter.parse(file)
    
    #grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    #Looping over all the instruments
    for part in s2.parts:
        notes_to_parse = part.recurse() 
        #finding whether a particular element is note or a chord
        for element in notes_to_parse:        
            #note
            if isinstance(element, note.Note):
                # Read only duration of the note
                notes.append(str(element.duration.quarterLength))        
            #chord
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    return np.array(notes)

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
def train():
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

def evaluate(eval_model, data_source):
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

    #%% DATA PREPARATION
    
    # LOAD DATASET
    dataset = ImprovDataset()
    X = dataset.getData()
    # set vocabulary for conversion
    vocab = dataset.vocab
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    print(X[:3])
    
    # Divide into train, validation and test
    train_notes = X[:int(len(X)*0.7)]
    val_notes = X[int(len(X)*0.7)+1:int(len(X)*0.7)+1+int(len(X)*0.1)]
    test_notes = X[int(len(X)*0.7)+1+int(len(X)*0.1):]
    
    # divide into batches of size bsz and converts notes into numbers
    def batchify(data, bsz):
        new_data = []
        for sequence in data:
            for note in sequence:
                new_data.append(note)
        new_data = np.array(new_data)
        
        data = torch.tensor([word_to_ix[w] for w in new_data], dtype=torch.long)
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)
    
    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_notes, batch_size)
    val_data = batchify(val_notes, eval_batch_size)
    test_data = batchify(test_notes, eval_batch_size)

    # divide into target and input sequence of lenght bptt
    # --> obtain matrices of size bptt x batch_size
    bptt = 35
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len] # input 
        target = source[i+1:i+1+seq_len].view(-1) # target (same as input but shifted by 1)
        return data, target
    
    #%% MODEL TRAINING
    
    # HYPERPARAMETERS
    ntokens = len(vocab) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 10 # The number of epochs
    best_model = None
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    
        scheduler.step()
    
    # TEST THE MODEL
    test_loss = evaluate(best_model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    
    #%% SAMPLES GENERATION

    
    def getNote(val): 
        for key, value in word_to_ix.items(): 
             if val == value: 
                 return key

    def generate(model, melody4gen, next_notes=10):
        for i in range(0,next_notes):
            x_pred = torch.tensor([word_to_ix[w] for w in melody4gen], dtype=torch.long)
            y_pred = model(x_pred)
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            melody4gen.append(getNote(word_index))
        return melody4gen

    melody4gen = test_notes[0][:10]
    new_melody = generate(model, melody4gen, 10)
    
    print(new_melody)
    
    
    n1 = note.Note()
    n1.pitch.nameWithOctave = 'E-5'
    n1.duration.quarterLength = 3.0
