#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:05:28 2021

@author: vincenzomadaghiele
"""
import sys
import argparse
import os
import shutil
import time
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import B_train.loadDB as dataset
import B_train.MINGUS_model as mod

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--COND_TYPE_PITCH', type=str, default='I-C-NC-B-BE-O',
                    help='conditioning features for pitch model')
    parser.add_argument('--COND_TYPE_DURATION', type=str, default='I-C-NC-B-BE-O',
                    help='conditioning features for duration model')
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=20,
                        help='training batch size')
    parser.add_argument('--EVAL_BATCH_SIZE', type=int, default=10,
                        help='evaluation batch size')
    parser.add_argument('--BPTT', type=int, default=35,
                    help='length of a note sequence for training')
    parser.add_argument('--EPOCHS', type=int, default=10,
                    help='epochs for training')
    parser.add_argument('--SEGMENTATION', action='store_false', default=True,
                        help='train with NO melody segmentation')
    parser.add_argument('--AUGMENTATION', action='store_true', default=False,
                        help='augment dataset')
    parser.add_argument('--augmentation_const', type=int, default=3,
                        help='how many times to augment the data')
    args = parser.parse_args(sys.argv[1:])
    
    # Constants for MINGUS training
    print('Training summary:')
    print('-' * 80)
    print('TRAIN_BATCH_SIZE:', args.TRAIN_BATCH_SIZE)
    print('EVAL_BATCH_SIZE:', args.EVAL_BATCH_SIZE)
    print('EPOCHS:', args.EPOCHS)
    print('sequence length:', args.BPTT)
    print('SEGMENTATION:', args.SEGMENTATION)
    print('AUGMENTATION:', args.AUGMENTATION) 
    print('augmentation_const:', args.augmentation_const)
    print('pitch model conditionings:', args.COND_TYPE_PITCH)
    print('duration model conditionings:', args.COND_TYPE_DURATION)
    print('-' * 80)
    
    # LOAD DATA
    
    MusicDB = dataset.MusicDB(device, args.TRAIN_BATCH_SIZE, args.EVAL_BATCH_SIZE,
                 args.BPTT, args.AUGMENTATION, args.SEGMENTATION, args.augmentation_const)
    
    vocabPitch, vocabDuration, vocabBeat, vocabOffset = MusicDB.getVocabs()
    
    pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = MusicDB.getInverseVocabs()
    
    train_pitch_batched, train_duration_batched, train_chord_batched, train_next_chord_batched, train_bass_batched, train_beat_batched, train_offset_batched  = MusicDB.getTrainingData()
    val_pitch_batched, val_duration_batched, val_chord_batched, val_next_chord_batched, val_bass_batched, val_beat_batched, val_offset_batched  = MusicDB.getValidationData()
    test_pitch_batched, test_duration_batched, test_chord_batched, test_next_chord_batched, test_bass_batched, test_beat_batched, test_offset_batched  = MusicDB.getTestData()


    #%% PITCH MODEL TRAINING
    
    isPitch = True
    # HYPERPARAMETERS
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 512
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 512
    
    chord_encod_dim = 64
    next_chord_encod_dim = 32

    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 64
    bass_embed_dim = 64
    
    offset_vocab_size = len(vocabOffset) # size of the duration vocabulary
    offset_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    offset_pad_idx = offset_to_ix['<pad>']
    modelPitch = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
                                      beat_vocab_size, beat_embed_dim,
                                      offset_vocab_size, offset_embed_dim,
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
                                      device, dropout, isPitch, args.COND_TYPE_PITCH).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss(ignore_index=pitch_pad_idx)
    lr = 0.5 # learning rate
    optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr, momentum=0.9,  nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = args.EPOCHS # The number of epochs
    best_model = None
    
    # INITIALIZE TENSORBOARD
    path = f'B_train/runs/pitchModel/COND {args.COND_TYPE_PITCH} Epochs {epochs} Augmentation {args.AUGMENTATION}'
    if os.path.isdir(path):
        # Remove folder with same parameters
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    step = 0

    pitch_start_time = time.time()
    # TRAINING LOOP
    print('Starting pitch model training...')
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        step = mod.train(modelPitch, vocabPitch, 
                  train_pitch_batched, train_duration_batched, 
                  train_chord_batched, train_next_chord_batched,
                  train_bass_batched, train_beat_batched, train_offset_batched,
                  criterion, optimizer, scheduler, epoch, args.BPTT, device, 
                  writer, step, 
                  isPitch)
        
        val_loss, val_acc = mod.evaluate(modelPitch, pitch_to_ix, 
                                val_pitch_batched, val_duration_batched, 
                                val_chord_batched, val_next_chord_batched,
                                val_bass_batched, val_beat_batched, val_offset_batched,
                                criterion, args.BPTT, device, isPitch)
        
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
                                test_pitch_batched, test_duration_batched, 
                                test_chord_batched, test_next_chord_batched,
                                test_bass_batched, test_beat_batched, test_offset_batched,
                                criterion, args.BPTT, device, isPitch)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:5.2f} | test acc {:5.2f}'.format(
        test_loss, math.exp(test_loss), test_acc))
    print('=' * 89)
    
    models_folder = "models"
    model_name = "MINGUSpitch"
    num_epochs = str(epochs) + "epochs"
    segm_len = "seqLen" + str(args.BPTT)
    savePATHpitch = (models_folder + '/' + model_name + '_' + num_epochs 
                     + '_'+ segm_len + '_' + '.pt')
    
    savePATHpitch = f'B_train/models/pitchModel/MINGUS COND {args.COND_TYPE_PITCH} Epochs {args.EPOCHS}.pt'
    
    state_dictPitch = best_model_pitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    writer.close()
    
    
    #%% DURATION MODEL TRAINING
    
    isPitch = False
    # HYPERPARAMETERS
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 64
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 64
    
    chord_encod_dim = 64
    next_chord_encod_dim = 32
    
    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 32
    bass_embed_dim = 32
    
    offset_vocab_size = len(vocabOffset) # size of the duration vocabulary
    offset_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    offset_pad_idx = offset_to_ix['<pad>']
    modelDuration = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
                                      beat_vocab_size, beat_embed_dim, 
                                      offset_vocab_size, offset_embed_dim,
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
                                      device, dropout, isPitch, args.COND_TYPE_DURATION).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss(ignore_index=duration_pad_idx)
    lr = 0.05 # learning rate
    optimizer = torch.optim.SGD(modelDuration.parameters(), lr=lr, momentum=0.9,  nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = args.EPOCHS # The number of epochs
    best_model = None

    # INITIALIZE TENSORBOARD
    path = f'B_train/runs/durationModel/COND {args.COND_TYPE_DURATION} EPOCHS {epochs} Augmentation {args.AUGMENTATION}'
    #path = f'runs/{con.DATASET}/durationModel/REDUCE LR ON PLATEAU'    
    if os.path.isdir(path):
        # Remove folder with same parameters
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    step = 0

    duration_start_time = time.time()
    # TRAINING LOOP
    print('Starting duration model training...')
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        step = mod.train(modelDuration, vocabDuration, 
                  train_pitch_batched, train_duration_batched, 
                  train_chord_batched, train_next_chord_batched,
                  train_bass_batched, train_beat_batched, train_offset_batched,
                  criterion, optimizer, scheduler, epoch, args.BPTT, device, 
                  writer, step,
                  isPitch)
        
        val_loss, val_acc = mod.evaluate(modelDuration, duration_to_ix, 
                                val_pitch_batched, val_duration_batched, 
                                val_chord_batched, val_next_chord_batched,
                                val_bass_batched, val_beat_batched, val_offset_batched,
                                criterion, args.BPTT, device, isPitch)
        
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
                                test_pitch_batched, test_duration_batched, 
                                test_chord_batched, test_next_chord_batched,
                                test_bass_batched, test_beat_batched, test_offset_batched,
                                criterion, args.BPTT, device, isPitch)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:5.2f} | test acc {:5.2f}'.format(
        test_loss, math.exp(test_loss), test_acc))
    print('=' * 89)
    
    models_folder = "models"
    model_name = "MINGUSduration"
    num_epochs = str(epochs) + "epochs"
    segm_len = "seqLen" + str(args.BPTT)
    savePATHduration = (models_folder + '/' + model_name + '_' + num_epochs 
                     + '_'+ segm_len + '_' + '.pt')
    
    savePATHduration = f'B_train/models/durationModel/MINGUS COND {args.COND_TYPE_DURATION} Epochs {args.EPOCHS}.pt'
    
    state_dictDuration = best_model_duration.state_dict()
    torch.save(state_dictDuration, savePATHduration)
    writer.close()
    
    
    print('Total training time for pitch model: ', pitch_end_time - pitch_start_time )
    print('Total training time for duration model: ', duration_end_time - duration_start_time)


