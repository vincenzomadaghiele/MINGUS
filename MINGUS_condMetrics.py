#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:33:11 2021

@author: vincenzomadaghiele
"""

import torch
import torch.nn as nn
import math
import time
import os
import shutil
import itertools
import json
import numpy as np
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
        
        train_pitch_batched, train_duration_batched, train_chord_batched, train_next_chord_batched, train_bass_batched, train_beat_batched, train_offset_batched  = WjazzDB.getTrainingData()
        val_pitch_batched, val_duration_batched, val_chord_batched, val_next_chord_batched, val_bass_batched, val_beat_batched, val_offset_batched  = WjazzDB.getValidationData()
        test_pitch_batched, test_duration_batched, test_chord_batched, test_next_chord_batched, test_bass_batched, test_beat_batched, test_offset_batched  = WjazzDB.getTestData()
        
        songs = WjazzDB.getOriginalSongDict()
        structuredSongs = WjazzDB.getStructuredSongs()
        vocabPitch, vocabDuration, vocabBeat, vocabOffset = WjazzDB.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = WjazzDB.getInverseVocabs()
        WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = WjazzDB.getChordDicts()

        
    elif con.DATASET == 'NottinghamDB':
        
        NottinghamDB = dataset.NottinghamDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                                            con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
                        
        train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = NottinghamDB.getTrainingData()
        val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = NottinghamDB.getValidationData()
        test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = NottinghamDB.getTestData()
    
        songs = NottinghamDB.getOriginalSongDict()
        structuredSongs = NottinghamDB.getStructuredSongs()
        vocabPitch, vocabDuration, vocabBeat = NottinghamDB.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix = NottinghamDB.getInverseVocabs()
        NottinghamChords, NottinghamToMusic21, NottinghamToChordComposition, NottinghamToMidiChords = NottinghamDB.getChordDicts()

    
    # number of conditioned models to consider
    NUM_MODELS = 5
    EPOCHS = 20
    
    #%% Opening JSON file of multiple models scores
    pitch_scores = open('scores/pitch.json',)
    pitch_scores = json.load(pitch_scores)
    duration_scores = open('scores/duration.json',)
    duration_scores = json.load(duration_scores)
    
    # find best conditionings by accuracy
    pitch_conds = []
    pitch_acc = []
    for cond, metrics in pitch_scores.items():
        pitch_conds.append(cond)
        pitch_acc.append(metrics['val acc'])
    
    pitch_acc = np.array(pitch_acc)
    pitch_conds = np.array(pitch_conds)

    sorted_index_pitch = np.argsort(pitch_acc)
    sorted_acc_pitch = pitch_acc[sorted_index_pitch]
    BEST_CONDS_PITCH = pitch_conds[sorted_index_pitch][::-1][:NUM_MODELS]
    
    duration_conds = []
    duration_acc = []
    for cond, metrics in duration_scores.items():
        duration_conds.append(cond)
        duration_acc.append(metrics['val acc'])
    
    duration_acc = np.array(duration_acc)
    duration_conds = np.array(duration_conds)

    sorted_index_duration = np.argsort(duration_acc)
    sorted_acc_duration = duration_acc[sorted_index_duration]
    BEST_CONDS_DURATION = duration_conds[sorted_index_duration][::-1][:NUM_MODELS]

    for COND_PITCH in BEST_CONDS_PITCH:
        for COND_DURATION in BEST_CONDS_DURATION:
            # LOAD PRE-TRAINED MODELS
    
            # PITCH MODEL
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
                                              device, dropout, isPitch, con.COND_TYPE_PITCH).to(device)
            
            if con.DATASET == 'WjazzDB':
                savePATHpitch = 'models/MINGUSpitch_10epochs_seqLen35_WjazzDB.pt'
                
                savePATHpitch = f'models/{con.DATASET}/pitchModel/MINGUS COND {COND_PITCH} Epochs {EPOCHS}.pt'
            
            elif con.DATASET == 'NottinghamDB':
                savePATHpitch = 'models/MINGUSpitch_100epochs_seqLen35_NottinghamDB.pt'
            modelPitch.load_state_dict(torch.load(savePATHpitch, map_location=torch.device('cpu')))
                
            
            # DURATION MODEL
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
                                              device, dropout, isPitch, con.COND_TYPE_DURATION).to(device)
            
            if con.DATASET == 'WjazzDB':
                savePATHduration = 'models/MINGUSduration_200epochs_seqLen35_WjazzDB.pt'
                
                savePATHduration = f'models/{con.DATASET}/durationModel/MINGUS COND {con.COND_TYPE_DURATION} Epochs {EPOCHS}.pt'
            
            elif con.DATASET == 'NottinghamDB':
                savePATHduration = 'models/MINGUSduration_100epochs_seqLen35_NottinghamDB.pt'
            modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))
                    
            
            # Generate new music
            # Set to True to generate dataset of songs
            generate_dataset = True
            
            if generate_dataset:
                out_path = 'output/gen4eval_' + con.DATASET + '/'
                generated_path = 'generated/'
                original_path = 'original/'
                num_tunes = 10
                generated_structuredSongs = []
                original_structuredSongs = []
        
                for tune in structuredSongs[:num_tunes]:
                    
                    if con.DATASET == 'WjazzDB':
                        isJazz = True
                        new_structured_song = generateCond(tune, num_bars, temperature, 
                                                   modelPitch, modelDuration, WjazzToMidiChords, isJazz)
                        
                        pm = structuredSongsToPM(new_structured_song, WjazzToMidiChords)
                        pm.write(out_path + generated_path + new_structured_song['title'] + '.mid')
                        pm = structuredSongsToPM(tune, WjazzToMidiChords)
                        pm.write(out_path + original_path + tune['title'] + '.mid')
                        generated_structuredSongs.append(new_structured_song)
                        original_structuredSongs.append(tune)
                        
                    elif con.DATASET == 'NottinghamDB':
                        isJazz = False
                        new_structured_song = generateCond(tune, num_bars, temperature, 
                                                   modelPitch, modelDuration, NottinghamToMidiChords)
                        
                        pm = structuredSongsToPM(new_structured_song, NottinghamToMidiChords)
                        pm.write(out_path + generated_path + new_structured_song['title'] + '.mid')
                        pm = structuredSongsToPM(tune, NottinghamToMidiChords)
                        pm.write(out_path + original_path + tune['title'] + '.mid')
                        generated_structuredSongs.append(new_structured_song)
                        original_structuredSongs.append(tune)
                    
                # Convert dict to JSON and SAVE IT
                with open(out_path + generated_path + con.DATASET + '_generated.json', 'w') as fp:
                    json.dump(generated_structuredSongs, fp, indent=4)
                with open(out_path + original_path + con.DATASET + '_original.json', 'w') as fp:
                    json.dump(original_structuredSongs, fp, indent=4)
            
            # Evaluate
            
        
