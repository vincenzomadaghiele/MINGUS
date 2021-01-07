#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:08:32 2021

@author: vincenzomadaghiele
"""
import pretty_midi
import torch
import torch.nn as nn
import numpy as np
import math
import time
import MINGUS_dataset_funct as dataset
import LSTM_model as mod

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':

    # DATA LOADING
    
    # LOAD PITCH DATASET
    
    dataset_folder = "data"
    dataset_name = "folkDB"
    pitch_path = dataset_folder +'/'+dataset_name+'/'
    
    datasetPitch = dataset.ImprovPitchDataset(pitch_path, 20)
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
    #duration_path = 'data/w_jazz/'
    duration_path = dataset_folder + '/' + dataset_name + '/'
    
    datasetDuration = dataset.ImprovDurationDataset(duration_path, 10)
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
    
    
    #%% DATA PRE-PROCESSING
    
    # Maximum value of a sequence
    segment_length = 32 # same as ECMG paper
    # Melody segmentation
    train_pitch_segmented, train_duration_segmented = dataset.segmentDataset(train_pitch, train_duration, segment_length)
    val_pitch_segmented, val_duration_segmented = dataset.segmentDataset(val_pitch, val_duration, segment_length)
    test_pitch_segmented, test_duration_segmented = dataset.segmentDataset(test_pitch, test_duration, segment_length)


    batch_size = 32
    eval_batch_size = 32
    # Batch the data
    train_data_pitch = dataset.batchify(train_pitch_segmented, batch_size, pitch_to_ix, device)
    val_data_pitch = dataset.batchify(val_pitch_segmented, eval_batch_size, pitch_to_ix, device)
    test_data_pitch = dataset.batchify(test_pitch_segmented, eval_batch_size, pitch_to_ix, device)
    
    train_data_duration = dataset.batchify(train_duration_segmented, batch_size, duration_to_ix, device)
    val_data_duration = dataset.batchify(val_duration_segmented, eval_batch_size, duration_to_ix, device)
    test_data_duration = dataset.batchify(test_duration_segmented, eval_batch_size, duration_to_ix, device)
    
    # divide into target and input sequence of lenght bptt
    # --> obtain matrices of size bptt x batch_size
    # a padded sequence is of length segment_value + 2 (sos and eos tokens)
    bptt = segment_length #+ 2 # lenght of a sequence of data (IMPROVEMENT HERE!!)
    
    
    #%% PITCH MODEL TRAINING
    
    
    # HYPERPARAMETERS
    src_pad_idx = pitch_to_ix['<pad>']

    # These values are set as the default used in the ECMG paper
    vocab_size = len(vocabPitch)
    embed_dim = 8 # 8 for pitch, 4 for duration
    # the dimension of the output is the same 
    # as the input because the vocab is the same
    output_dim = len(vocabPitch)
    hidden_dim = 256
    seq_len = 32
    num_layers = 2
    batch_size = 32
    dropout = 0.5
    batch_norm=True # ECMG original is true
    no_cuda=False

    modelPitch = mod.NoCondLSTM(vocab_size, embed_dim, output_dim, 
                                hidden_dim, seq_len, num_layers, batch_size, 
                                dropout, batch_norm, no_cuda).to(device)
    
    # LOSS FUNCTION
    criterion = nn.NLLLoss(ignore_index=src_pad_idx)
    lr = 1e-3 # default learning rate of ECMG code
    optimizer = torch.optim.Adam(modelPitch.parameters(), lr=lr, amsgrad=True)
   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) # not in ECMG code
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 10 # The number of epochs
    best_model = None

    
    # TRAINING LOOP
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        mod.train(modelPitch, vocabPitch, train_data_pitch, criterion, optimizer, 
              scheduler, epoch, bptt)
        val_loss = mod.evaluate(modelPitch, val_data_pitch, vocabPitch, 
                            criterion, bptt, device)
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
    test_loss = mod.evaluate(best_model_pitch, test_data_pitch, vocabPitch, 
                         criterion, bptt, device)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    
    
    models_folder = "models"
    model_name = "LSTMpitch"
    num_epochs = str(epochs) + "epochs"
    segm_len = "seqLen" + str(segment_length)
    savePATHpitch = (models_folder + '/' + model_name + '_' + num_epochs 
                     + '_'+ segm_len + '_' + dataset_name + '.pt')
    
    state_dictPitch = best_model_pitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    

    #%% DURATION MODEL TRAINING
    
    # HYPERPARAMETERS
    src_pad_idx = pitch_to_ix['<pad>']

    # These values are set as the default used in the ECMG paper
    vocab_size = len(vocabDuration)
    embed_dim = 4 # 8 for pitch, 4 for duration
    # the dimension of the output is the same 
    # as the input because the vocab is the same
    output_dim = len(vocabDuration)
    hidden_dim = 256
    seq_len = 32
    num_layers = 2
    batch_size = 32
    dropout = 0.5
    batch_norm = True # ECMG original is true
    no_cuda = False

    modelDuration = mod.NoCondLSTM(vocab_size, embed_dim, output_dim, 
                                hidden_dim, seq_len, num_layers, batch_size, 
                                dropout, batch_norm, no_cuda).to(device)
    
    # LOSS FUNCTION
    criterion = nn.NLLLoss(ignore_index=src_pad_idx)
    lr = 1e-3 # default learning rate of ECMG code
    optimizer = torch.optim.Adam(modelPitch.parameters(), lr=lr, amsgrad=True)
   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) # not in ECMG code
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 10 # The number of epochs
    best_model = None

    
    # TRAINING LOOP
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        mod.train(modelDuration, vocabDuration, train_data_duration, criterion, optimizer, 
              scheduler, epoch, bptt)
        val_loss = mod.evaluate(modelDuration, val_data_duration, vocabDuration, 
                            criterion, bptt, device)
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
    test_loss = mod.evaluate(best_model_duration, test_data_duration, vocabDuration, 
                         criterion, bptt, device)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    
    
    models_folder = "models"
    model_name = "LSTMduration"
    num_epochs = str(epochs) + "epochs"
    segm_len = "seqLen" + str(segment_length)
    savePATHduration = (models_folder + '/' + model_name + '_' + num_epochs 
                     + '_'+ segm_len + '_' + dataset_name + '.pt')
    
    state_dictPitch = best_model_duration.state_dict()
    torch.save(state_dictPitch, savePATHduration)
            
            