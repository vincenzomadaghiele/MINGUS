#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 08:56:16 2021

@author: vincenzomadaghiele

ToDo:
    Done:
        - Tensorboard
        - Audio rendition
        - Wjazz generations
        
    To ask:
        - robinson not working
        - how to optimize model
            change transformer layer (?)
            reduce dictionary (?)
            quantize durations (?)
    
    Model:
        - change dimensions for pitch and duration model
        - modular model to compare training with or without parameters
        - fine tune parameters for NottinghamDB training 
        - fine tune parameters for WjazzDB training 
        - include offset (maybe it improves duration inference)

    Evaluation:
        - generation/evaluation on EURECOM machine
        - export midi to audio and evaluate long-term coherence 
            with Jazz Transformer metric (Matlab)

    Future steps:
        - bass line generation on WjazzDB (given only chord and given melody)
        - parallel conditional generation (melody + bass)
        - consider note_seq pipeline with quantization to obtain better results on WjazzDB
        - evaluate implementation of Transformer-XL / Relative attention (and comparison)
        - live interface (chord recognition in max from audio or midi), 
            note generation in python (communication via osc)
        might require training with different bass embedding layer
        - melody harmonization

    FolkDB:
        - check if in midi folkDB files there is chords information with note_seq
        - if no chord information on folkDB try to open abc files with other library


Training instruction:
    - to train with different dataset create a json file in the same format 
        of the three given song datasets:
            divided in train, validation, test and structured songs
            give clear information about the format
        different datasets can be automatically transcribed with the functions provided
    - to train with the given datasets just modify the MINGUS_const file,
        which contains the training parameters
    - do not train with structured song data:
        the rests duration are not the same! 
        OR take data from structured song but implement function to 'sum' rests
    
"""

import torch
import torch.nn as nn
import math
import time
import os
import shutil
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':

    # LOAD DATA
    
    if con.DATASET == 'WjazzDB':
        
        WjazzDB = dataset.WjazzDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                     con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
        
        vocabPitch, vocabDuration, vocabBeat = WjazzDB.getVocabs()
        
        pitch_to_ix, duration_to_ix, beat_to_ix = WjazzDB.getInverseVocabs()
        
        train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
        val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getValidationData()
        test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTestData()
        
    elif con.DATASET == 'NottinghamDB':
        
        NottinghamDB = dataset.NottinghamDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                                            con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
        
        vocabPitch, vocabDuration, vocabBeat = NottinghamDB.getVocabs()
        
        pitch_to_ix, duration_to_ix, beat_to_ix = NottinghamDB.getInverseVocabs()
        
        train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = NottinghamDB.getTrainingData()
        val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = NottinghamDB.getValidationData()
        test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = NottinghamDB.getTestData()
    
    
    #%% PITCH MODEL TRAINING
    
    isPitch = True
    # HYPERPARAMETERS
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 512
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 512
    
    chord_encod_dim = 64

    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 64
    bass_embed_dim = 64


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
                                      bass_embed_dim, chord_encod_dim,
                                      beat_vocab_size, beat_embed_dim,  
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx,
                                      device, dropout, isPitch).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss(ignore_index=pitch_pad_idx)
    lr = 0.5 # learning rate
    optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr, momentum=0.9,  nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = con.EPOCHS # The number of epochs
    best_model = None
    
    # INITIALIZE TENSORBOARD
    path = f'runs/{con.DATASET}/pitchModel/Epochs {epochs} LR {lr} BPTT {con.BPTT} BatchSize {con.TRAIN_BATCH_SIZE} Augmentation {con.AUGMENTATION} Segmentation {con.SEGMENTATION}'
    #path = f'runs/{con.DATASET}/pitchModel/REDUCE LR ON PLATEAU'
    if os.path.isdir(path):
        # Remove folder with same parameters
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    step = 0

    pitch_start_time = time.time()
    # TRAINING LOOP
    print('Starting training...')
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        step = mod.train(modelPitch, vocabPitch, 
                  train_pitch_batched, train_duration_batched, train_chord_batched,
                  train_bass_batched, train_beat_batched,
                  criterion, optimizer, scheduler, epoch, con.BPTT, device, 
                  writer, step, 
                  isPitch)
        
        val_loss, val_acc = mod.evaluate(modelPitch, pitch_to_ix, 
                                val_pitch_batched, val_duration_batched, val_chord_batched,
                                val_bass_batched, val_beat_batched,
                                criterion, con.BPTT, device, isPitch)
        
        writer.add_scalar('Validation accuracy', val_acc, global_step=epoch)
        
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
    
    dataset_name = con.DATASET
    models_folder = "models"
    model_name = "MINGUSpitch"
    num_epochs = str(epochs) + "epochs"
    segm_len = "seqLen" + str(con.BPTT)
    savePATHpitch = (models_folder + '/' + model_name + '_' + num_epochs 
                     + '_'+ segm_len + '_' + dataset_name + '.pt')
    
    state_dictPitch = best_model_pitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    #writer.close()
    
    
    #%% DURATION MODEL TRAINING
    
    isPitch = False
    # HYPERPARAMETERS
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 512
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 512
    
    chord_encod_dim = 64
    
    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 64
    bass_embed_dim = 64


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
                                      bass_embed_dim, chord_encod_dim,
                                      beat_vocab_size, beat_embed_dim,  
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx,
                                      device, dropout, isPitch).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss(ignore_index=duration_pad_idx)
    lr = 0.05 # learning rate
    optimizer = torch.optim.SGD(modelDuration.parameters(), lr=lr, momentum=0.9,  nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = con.EPOCHS # The number of epochs
    best_model = None

    # INITIALIZE TENSORBOARD
    path = f'runs/{con.DATASET}/durationModel/EPOCHS {epochs} LR {lr} BPTT {con.BPTT} BatchSize {con.TRAIN_BATCH_SIZE} Augmentation {con.AUGMENTATION} Segmentation {con.SEGMENTATION}'
    #path = f'runs/{con.DATASET}/durationModel/REDUCE LR ON PLATEAU'    
    if os.path.isdir(path):
        # Remove folder with same parameters
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    step = 0

    duration_start_time = time.time()
    # TRAINING LOOP
    print('Starting training...')
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        step = mod.train(modelDuration, vocabDuration, 
                  train_pitch_batched, train_duration_batched, train_chord_batched,
                  train_bass_batched, train_beat_batched,
                  criterion, optimizer, scheduler, epoch, con.BPTT, device, 
                  writer, step,
                  isPitch)
        
        val_loss, val_acc = mod.evaluate(modelDuration, duration_to_ix, 
                                val_pitch_batched, val_duration_batched, val_chord_batched,
                                val_bass_batched, val_beat_batched,
                                criterion, con.BPTT, device, isPitch)
        
        writer.add_scalar('Validation accuracy', val_acc, global_step=epoch)
        
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
    
    dataset_name = con.DATASET
    models_folder = "models"
    model_name = "MINGUSduration"
    num_epochs = str(epochs) + "epochs"
    segm_len = "seqLen" + str(con.BPTT)
    savePATHpitch = (models_folder + '/' + model_name + '_' + num_epochs 
                     + '_'+ segm_len + '_' + dataset_name + '.pt')
    
    state_dictPitch = best_model_duration.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    #writer.close()
    
    
    print('Total training time for pitch model: ', pitch_end_time - pitch_start_time )
    print('Total training time for duration model: ', duration_end_time - duration_start_time)
    
    
