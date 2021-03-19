#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 08:56:16 2021

@author: vincenzomadaghiele
"""

import torch
import torch.nn as nn
import math
import time
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':

    # LOAD DATA 
    
    WjazzDB = dataset.WjazzDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                 con.BPTT, con.AUGMENTATION, con.SEGMENTATION)
    
    vocabPitch, vocabDuration, vocabBeat = WjazzDB.getVocabs()
    
    pitch_to_ix, duration_to_ix, beat_to_ix = WjazzDB.getInverseVocabs()
    
    train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
    val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getTrainingData()
    test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTrainingData()
    
    
    #%% PITCH MODEL TRAINING
    
    isPitch = True
    # HYPERPARAMETERS
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 64
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 64
    
    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 32
    bass_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    modelPitch = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, 
                                      beat_vocab_size, beat_embed_dim,  
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx,
                                      device, dropout, isPitch).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss(ignore_index=pitch_pad_idx)
    lr = 0.5 # learning rate
    optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 10 # The number of epochs
    best_model = None

    pitch_start_time = time.time()
    # TRAINING LOOP
    print('Starting training...')
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        mod.train(modelPitch, pitch_to_ix, 
                  train_pitch_batched, train_duration_batched, train_chord_batched,
                  train_bass_batched, train_beat_batched,
                  criterion, optimizer, scheduler, epoch, con.BPTT, device, isPitch)
        
        val_loss, val_acc = mod.evaluate(modelPitch, pitch_to_ix, 
                                val_pitch_batched, val_duration_batched, val_chord_batched,
                                val_bass_batched, val_beat_batched,
                                criterion, con.BPTT, device, isPitch)
        
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:5.2f} | valid acc {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss), val_acc))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_pitch = modelPitch
    
        scheduler.step()
    
    pitch_end_time = time.time()
    
    
    # TEST THE MODEL
    test_loss, test_acc = mod.evaluate(best_model_pitch, pitch_to_ix, 
                                test_pitch_batched, test_duration_batched, test_chord_batched, 
                                test_bass_batched, test_beat_batched,
                                criterion, con.BPTT, device, isPitch)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:5.2f} | test acc {:5.2f}'.format(
        test_loss, math.exp(test_loss), test_acc))
    print('=' * 89)
    
    dataset_name = 'Wjazz'
    models_folder = "models"
    model_name = "MINGUSpitch"
    num_epochs = str(epochs) + "epochs"
    segm_len = "seqLen" + str(con.BPTT)
    savePATHpitch = (models_folder + '/' + model_name + '_' + num_epochs 
                     + '_'+ segm_len + '_' + dataset_name + '.pt')
    
    state_dictPitch = best_model_pitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    
    
    #%% DURATION MODEL TRAINING
    
    isPitch = False
    # HYPERPARAMETERS
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 64
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 64
    
    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 32
    bass_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    modelDuration = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, 
                                      beat_vocab_size, beat_embed_dim,  
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx,
                                      device, dropout, isPitch).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss(ignore_index=duration_pad_idx)
    lr = 0.5 # learning rate
    optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 10 # The number of epochs
    best_model = None

    duration_start_time = time.time()
    # TRAINING LOOP
    print('Starting training...')
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        mod.train(modelDuration, duration_to_ix, 
                  train_pitch_batched, train_duration_batched, train_chord_batched,
                  train_bass_batched, train_beat_batched,
                  criterion, optimizer, scheduler, epoch, con.BPTT, device, isPitch)
        
        val_loss, val_acc = mod.evaluate(modelDuration, duration_to_ix, 
                                val_pitch_batched, val_duration_batched, val_chord_batched,
                                val_bass_batched, val_beat_batched,
                                criterion, con.BPTT, device, isPitch)
        
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:5.2f} | valid acc {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss), val_acc))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_duration = modelDuration
    
        scheduler.step()
    
    duration_end_time = time.time()
    
    
    # TEST THE MODEL
    test_loss, test_acc = mod.evaluate(best_model_duration, duration_to_ix, 
                                test_pitch_batched, test_duration_batched, test_chord_batched,
                                test_bass_batched, test_beat_batched,
                                criterion, con.BPTT, device, isPitch)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:5.2f} | test acc {:5.2f}'.format(
        test_loss, math.exp(test_loss), test_acc))
    print('=' * 89)
    
    dataset_name = 'Wjazz'
    models_folder = "models"
    model_name = "MINGUSduration"
    num_epochs = str(epochs) + "epochs"
    segm_len = "seqLen" + str(con.BPTT)
    savePATHpitch = (models_folder + '/' + model_name + '_' + num_epochs 
                     + '_'+ segm_len + '_' + dataset_name + '.pt')
    
    state_dictPitch = best_model_duration.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    
    print('Total training time for pitch model: ', pitch_end_time - pitch_start_time )
    print('Total training time for duration model: ', duration_end_time - duration_start_time)
    
    