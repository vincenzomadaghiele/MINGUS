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
    - Solve <pad> masking problem
    - implement remaining metrics (BLEU, MGEval)
    - run model with other datasets (seAttn data) and compare metrics
    - make net scheme
    - grid search for model optimization
    - conditioning on chords and inter-conditioning between pitch and duration
    - move all constants to an external .py file
    
Next meeting:
    - eurecom server problem with lengths of batches and pad mask
    - discuss real accuracy of the model
"""

import pretty_midi
import torch
import torch.nn as nn
import numpy as np
import math
from MINGUS_dataset_funct import ImprovDurationDataset, ImprovPitchDataset, readMIDI, convertMIDI

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

# TRANSFORMER MODEL
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, src_pad_idx, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.src_pad_idx = src_pad_idx
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_src_pad_mask(self, src):
        pad_mask = src.transpose(0, 1) == self.src_pad_idx
        #pad_mask = pad_mask.float().masked_fill(pad_mask == True, float('-inf')).masked_fill(pad_mask == False, float(0.0))
        return pad_mask.to(device)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src_padding_mask = self.make_src_pad_mask(src)
        #print(src_padding_mask)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_padding_mask)
        #output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        #print(output)
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
                    epoch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source, vocab):
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
    pitch_path = 'data/w_jazz/'
    datasetPitch = ImprovPitchDataset(pitch_path, 20)
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
    duration_path = 'data/w_jazz/'
    datasetDuration = ImprovDurationDataset(duration_path, 10)
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
    
    #%% Melody Segmentation
    
    def separateSeqs(seq_pitch, seq_duration, segment_length = 35):
        # Separate the songs into single melodies in order to avoid 
        # full batches of pad tokens
        
        tot_pitch = []
        tot_duration = []
        new_pitch = []
        new_duration = []
        long_dur = ['full', 'half', 'quarter', 'dot half', 'dot quarter', 
                    'dot 8th', 'half note triplet', 'quarter note triplet']
        counter = 0
        for i in range(min(len(seq_pitch), len(seq_duration))):
            new_pitch.append(seq_pitch[i])
            new_duration.append(seq_duration[i])
            counter += 1
            if seq_pitch[i] == 'R' and seq_duration[i] in long_dur:
                tot_pitch.append(np.array(new_pitch, dtype=object))
                tot_duration.append(np.array(new_duration, dtype=object))
                new_pitch = []
                new_duration = []
                counter = 0
            elif counter == segment_length:
                tot_pitch.append(np.array(new_pitch, dtype=object))
                tot_duration.append(np.array(new_duration, dtype=object))
                new_pitch = []
                new_duration = []
                counter = 0
        return tot_pitch, tot_duration
    
    def segmentDataset(pitch_data, duration_data, segment_length = 35):
        pitch_segmented = []
        duration_segmented = []
        for i in range(min(len(pitch_data), len(duration_data))):
            train_pitch_sep, train_duration_sep = separateSeqs(pitch_data[i], duration_data[i], segment_length)
            for seq in train_pitch_sep:
                pitch_segmented.append(seq)
            for seq in train_duration_sep:
                duration_segmented.append(seq)
        pitch_segmented = np.array(pitch_segmented, dtype=object)
        duration_segmented = np.array(duration_segmented, dtype=object)
        
        return pitch_segmented, duration_segmented
    
    # Maximum value of a sequence
    segment_length = 100
    train_pitch_segmented, train_duration_segmented = segmentDataset(train_pitch, train_duration, segment_length)
    val_pitch_segmented, val_duration_segmented = segmentDataset(val_pitch, val_duration, segment_length)
    test_pitch_segmented, test_duration_segmented = segmentDataset(test_pitch, test_duration, segment_length)
    
    
    
    #%% DATA PREPARATION

    # pad data to max_lenght of sequences, prepend <sos> and append <eos>
    def pad(data):
        '''

        Parameters
        ----------
        data : numpy ndarray or list
            list of sequences to be padded.

        Returns
        -------
        padded : numpy ndarray or list
            list of padded sequences.

        '''
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
               #([init_token])
                #+ 
                list(x[:max_len])
                #+ ([eos_token])
                + [pad_token] * max(0, max_len - len(x)))
        lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        return padded
    
    # divide into batches of size bsz and converts notes into numbers
    def batchify(data, bsz, dict_to_ix):
        '''

        Parameters
        ----------
        data : numpy ndarray or list
            data to be batched.
        bsz : int
            size of a batch.
        dict_to_ix : python dictionary
            dictionary for tokenization of the notes.

        Returns
        -------
        pytorch Tensor
            Data tokenized and divided in batches of bsz elements.

        '''
        
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
    
    train_data_pitch = batchify(train_pitch_segmented, batch_size, pitch_to_ix)
    val_data_pitch = batchify(val_pitch_segmented, eval_batch_size, pitch_to_ix)
    test_data_pitch = batchify(test_pitch_segmented, eval_batch_size, pitch_to_ix)
    
    train_data_duration = batchify(train_duration_segmented, batch_size, duration_to_ix)
    val_data_duration = batchify(val_duration_segmented, eval_batch_size, duration_to_ix)
    test_data_duration = batchify(test_duration_segmented, eval_batch_size, duration_to_ix)
    
    # divide into target and input sequence of lenght bptt
    # --> obtain matrices of size bptt x batch_size
    # a padded sequence is of length segment_value + 2 (sos and eos tokens)
    bptt = segment_length #+ 2 # lenght of a sequence of data (IMPROVEMENT HERE!!)
    def get_batch(source, i):
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
        return data, target
    
    
    #%% PITCH MODEL TRAINING
    
    # HYPERPARAMETERS
    ntokens_pitch = len(vocabPitch) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    src_pad_idx = pitch_to_ix['<pad>']
    modelPitch = TransformerModel(ntokens_pitch, emsize, nhead, nhid, nlayers, src_pad_idx, dropout).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 50 # The number of epochs
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
    
    savePATHpitch = 'modelsPitch/modelPitch_'+ str(epochs) + 'epochs_wjazz_segmented.pt'
    state_dictPitch = best_model_pitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    

    #%% DURATION MODEL TRAINING
    
    # HYPERPARAMETERS
    ntokens_duration = len(vocabDuration) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    src_pad_idx = duration_to_ix['<pad>']
    modelDuration = TransformerModel(ntokens_duration, emsize, nhead, nhid, nlayers, src_pad_idx, dropout).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(modelDuration.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 50 # The number of epochs
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
    
    savePATHduration = 'modelsDuration/modelDuration_'+ str(epochs) + 'epochs_wjazz_segmented.pt'
    state_dictDuration = best_model_duration.state_dict()
    torch.save(state_dictDuration, savePATHduration)
    

    
    #%% SAMPLES GENERATION

    '''
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
    
    # Remove characters who are not in the dictionary
    def onlyDict(pitchs, durations, vocabPitch, vocabDuration):
        # takes an array and a dictionary and gives the same array without
        # the elements who are not in the dictionary
        new_pitch = []
        new_duration = []
        for i in range(len(pitchs)):
            if pitchs[i] in vocabPitch and durations[i] in vocabDuration:
                new_pitch.append(pitchs[i]) 
                new_duration.append(durations[i]) 
        new_pitch = np.array(new_pitch) # same solos but with only most frequent notes
        new_duration = np.array(new_duration) # same solos but with only most frequent notes
        return new_pitch, new_duration
    
    
    #specify the path
    f = 'data/w_jazz/JohnColtrane_Mr.P.C._FINAL.mid'
    melody4gen_pitch, melody4gen_duration, dur_dict, song_properties = readMIDI(f)
    melody4gen_pitch, melody4gen_duration = onlyDict(melody4gen_pitch, melody4gen_duration, vocabPitch, vocabDuration)
    melody4gen_pitch = melody4gen_pitch[:80]
    melody4gen_duration = melody4gen_duration[:80]
    #print(melody4gen_pitch)
    #print(melody4gen_duration)
    
    notes2gen = 40 # number of new notes to generate
    new_melody_pitch = generate(modelPitch, melody4gen_pitch, pitch_to_ix, next_notes=notes2gen)
    new_melody_duration = generate(modelDuration, melody4gen_duration, duration_to_ix, notes2gen)

    converted = convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
    converted.write('output/music.mid')
    
    # For plotting
    import librosa.display
    import matplotlib.pyplot as plt
    def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
        # Use librosa's specshow function for displaying the piano roll
        librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                                 hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                                 fmin=pretty_midi.note_number_to_hz(start_pitch))
    
    plt.figure(figsize=(8, 4))
    plot_piano_roll(converted, 0, 127)    

    '''

    