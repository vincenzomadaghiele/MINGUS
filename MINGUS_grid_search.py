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

"""

import pretty_midi
import torch
import torch.nn as nn

import sklearn
import skorch
from skorch.utils import params_for

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
        return pad_mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        #src_padding_mask = self.make_src_pad_mask(src)
        #print(src_padding_mask)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        #output = self.transformer_encoder(src, src_mask, src_padding_mask)
        #print(output)
        output = self.transformer_encoder(src, src_mask)
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
    src_pad_idx = pitch_to_ix['<pad>']
    modelPitch = TransformerModel(ntokens_pitch, emsize, nhead, nhid, nlayers, src_pad_idx, dropout).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 2 # The number of epochs
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
    
    savePATHpitch = 'modelsPitch/modelPitch_'+ str(epochs) + 'epochs_EURECOM_augmented.pt'
    state_dictPitch = best_model_pitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    

    #%% DURATION MODEL TRAINING
    
    # HYPERPARAMETERS
    ntokens_duration = len(vocabDuration) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
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
    epochs = 2 # The number of epochs
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
    
    savePATHduration = 'modelsDuration/modelDuration_'+ str(epochs) + 'epochs_EURECOM_augmented.pt'
    state_dictDuration = best_model_duration.state_dict()
    torch.save(state_dictDuration, savePATHduration)
    
    
    #%% Grid Search
    # perform grid search using skorch
    
    
    class Trainer(skorch.NeuralNet):
        # override of the skorch NeuralNet class to adapt it to my Transformer
    
        def __init__(
            self, 
            *args, 
            **kwargs
        ):
            super().__init__(*args, **kwargs)
    
        def train_step(self, data, targets):
            # Pass here the results of get_batch
            self.module_.train()
            self.optimizer_.zero_grad()
            
            ntokens = len(vocabDuration) # Change with Pitch
            if data.size(0) != bptt:
                src_mask = self.module_.generate_square_subsequent_mask(data.size(0)).to(device)
    
            output = self.infer(data, src_mask)
            loss = self.get_loss(output.view(-1, ntokens), targets.reshape(-1), X=data, training=True)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.module_.parameters(), 0.5)
            self.optimizer_.step()
            
            return {'loss': loss, 'y_pred': output}
    
        def infer(self, Xi, yi=None):
            # Xi and yi are already tensors in my implementation 
            # (should work ??)
            return self.module_(Xi, yi)
        
        def get_loss(self, y_pred, y_true, **kwargs):
            #y_true = y_true[:, :y_pred.size(1)]        
            #y_pred_flat = y_pred.view(y_pred.size(0) * y_pred.size(1), -1)
            #y_true_flat = y_true.view(y_true.size(0) * y_true.size(1))
            
            return super().get_loss(
                y_pred,
                y_true,
                **kwargs)
        
        def _predict(self, X, most_probable=True):
            # return either predicted word probabilities or the most probable 
            # word using argmax.
            y_probas = []
            for yp in self.forward_iter(X, training=False):
                # yp is the result of a forward 
                # [bptt x batch_size x len(vocab)]
                if most_probable:
                    flat_durations = []
                    
                    # !! VERY INEFFICENT
                    for k in range(yp.shape[1]):
                        word_logits = yp[:,k]
                        for j in range(len(word_logits)):
                            logit = word_logits[j]
                            p = torch.nn.functional.softmax(logit, dim=0).detach().numpy()
                            word_index = np.random.choice(len(logit), p=p)
                            flat_durations.append(word_index)
                else:
                    flat_durations = []
                    for k in range(yp.shape[1]):
                        word_logits = yp[:,k]
                        flat_durations.append(word_logits)
                
                flat_durations = np.array(flat_durations)
                y_probas.append(flat_durations)
            y_proba = np.concatenate(y_probas, 0)
            return y_proba
        
        def predict_proba(self, X):
            return self._predict(X, most_probable=False)
        
        def predict(self, X):
            return self._predict(X, most_probable=True)
        
        def score(self, X, y):
            y_pred = self.predict(X)
            #y_true = y_pred.copy()
            
            # flatten y as the output of predict
            flat_y = []
            for batch in y:
                for k in range(batch.shape[1]):
                    for j in range(batch.shape[0]):
                        flat_y.append(batch[j,k])
            y_true = np.array(flat_y)
            
            return sklearn.metrics.accuracy_score(y_true.flatten(), y_pred.flatten())
    
    trainer = Trainer(
        criterion=torch.nn.NLLLoss,
        
        # We extended the trainer to support two optimizers
        # but to get the behavior of one optimizer we can 
        # simply use SGD for both, just like in the tutorial.
        optimizer_ = torch.optim.SGD,
        optimizer__lr = 5.0,
        
        module=TransformerModel,
        module__ntokens_duration = len(vocabDuration), 
        module__emsize = 200, 
        module__nhid = 200, 
        module__nlayers = 2, 
        module__nhead = 2, 
        module__dropout = 0.2, 
        module__src_pad_idx = duration_to_ix['<pad>'],

        
        # We have no internal validation.
        train_split=None,
        
        # The decoding code is not meant to be batched
        # so we have to deal with a batch size of 1 for
        # both training and validation/prediction.
        iterator_train__batch_size=1,
        iterator_valid__batch_size=1,
        
        # We are training only one large epoch.
        max_epochs=1,
        
        # Training takes a long time, add a progress bar
        # to see how far in we are. Since we are doing 
        # grid search with cross-validation splits and the
        # total amount of batches_per_epoch varies, we set
        # the batches_per_epoch method to 'auto', instead 
        # of the default, 'count'.
        callbacks=[
            skorch.callbacks.ProgressBar(batches_per_epoch='auto'),
        ],
        
        device=device,
    )
    
    bptt = 35 # lenght of a sequence of data (IMPROVEMENT HERE!!)
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len] # input 
        target = source[i+1:i+1+seq_len].reshape(-1) # target (same as input but shifted by 1)
        targets_no_reshape = source[i+1:i+1+seq_len]
        return data, target, targets_no_reshape
    
    # list of tensors
    X = []
    y = []
    for batch, i in enumerate(range(0, train_data_duration.size(0) - 1, bptt)):
        batch, _, target = get_batch(train_data_duration, i)
        # these are tensors 
        X.append(batch)
        y.append(target)
    
    # array of tensors
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=object)
    
    
    from sklearn.model_selection import GridSearchCV
    # change these parameters to good ones
    params = {
        'nhead': [4, 8],
        'nlayers': [4, 8],
    }
    gs = GridSearchCV(trainer, params)
    gs.fit(X, y)



    