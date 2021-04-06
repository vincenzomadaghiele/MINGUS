#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:15:44 2021

@author: vincenzomadaghiele
"""

import json
import os
import numpy as np
import math
import music21 as m21
import torch
import torch.nn as nn
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con
import MINGUS_eval_funct as ev

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
        
        
    #%% LOAD PRE-TRAINED MODELS
    
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
        
        savePATHpitch = f'models/{con.DATASET}/pitchModel/MINGUS COND {con.COND_TYPE_PITCH} Epochs {con.EPOCHS}.pt'
    
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
        savePATHduration = 'models/MINGUSduration_10epochs_seqLen35_WjazzDB.pt'
        
        savePATHduration = f'models/{con.DATASET}/durationModel/MINGUS COND {con.COND_TYPE_PITCH} Epochs {con.EPOCHS}.pt'
    
    elif con.DATASET == 'NottinghamDB':
        savePATHduration = 'models/MINGUSduration_100epochs_seqLen35_NottinghamDB.pt'
    modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))

    
    #%% Load generated structured songs
    
    gen_common_path = 'output/gen4eval_' + con.DATASET + '/'
    original_subpath = 'original/'
    generated_subpath = 'generated/'
    
    path = gen_common_path + original_subpath + con.DATASET + '_original.json'
    with open(path) as f:
        original_structuredSongs = json.load(f)
        
    path = gen_common_path + generated_subpath + con.DATASET + '_generated.json'
    with open(path) as f:
        generated_structuredSongs = json.load(f)
        
    
    #%% METRICS DICTIONARY
    
    # Make directory to save metrics file
    parent_directory = 'metrics/conditional_models/'
    model_id = str(con.EPOCHS) + 'epochs_' + str(con.BPTT) + 'seqlen_' + con.DATASET 
    path = os.path.join(parent_directory, model_id + '/')
    if not os.path.isdir(path):
        os.mkdir(path)
    
    # Instanciate dictionary
    metrics_result = {}
    metrics_result['MGEval'] = {}
    metrics_result['Harmonic coherence'] = {}
    metrics_result['Pitch'] = {}
    metrics_result['Duration'] = {}
    metrics_result['Pitch']['Pitch_accuracy'] = {}
    metrics_result['Pitch']['Pitch_perplexity'] = {}
    metrics_result['Pitch']['Pitch_test-loss'] = {}
    metrics_result['Pitch']['Pitch_BLEU'] = {}
    metrics_result['Duration']['Duration_accuracy'] = {}
    metrics_result['Duration']['Duration_perplexity'] = {}
    metrics_result['Duration']['Duration_test-loss'] = {}
    metrics_result['Duration']['Duration_BLEU'] = {}
    
    # MGEval on generated midi files
    num_of_generations = 10
    original_path = 'output/gen4eval_' + con.DATASET + '/original/*.mid'
    generated_path = 'output/gen4eval_' + con.DATASET + '/generated/*.mid'
    MGEresults = ev.MGEval(original_path, generated_path, path, num_of_generations)
    metrics_result['MGEval'] = MGEresults
    
    # Harmonic coherence of generated samples
    if con.DATASET == 'WjazzDB':
        original_scale_coherence, original_chord_coherence = ev.HarmonicCoherence(original_structuredSongs, 
                                                                                  WjazzToMusic21, 
                                                                                  WjazzToMidiChords)
        generated_scale_coherence, generated_chord_coherence = ev.HarmonicCoherence(generated_structuredSongs, 
                                                                                    WjazzToMusic21, 
                                                                                    WjazzToMidiChords)
    elif con.DATASET == 'NottinghamDB':
        original_scale_coherence, original_chord_coherence = ev.HarmonicCoherence(original_structuredSongs, 
                                                                                  NottinghamToMusic21, 
                                                                                  NottinghamToMidiChords)
        generated_scale_coherence, generated_chord_coherence = ev.HarmonicCoherence(generated_structuredSongs, 
                                                                                    NottinghamToMusic21, 
                                                                                    NottinghamToMidiChords)

    metrics_result['Harmonic coherence']['Original scale coherence'] = np.round_(original_scale_coherence, decimals=4)
    metrics_result['Harmonic coherence']['Original chord coherence'] = np.round_(original_chord_coherence, decimals=4)
    metrics_result['Harmonic coherence']['Generated scale coherence'] = np.round_(generated_scale_coherence, decimals=4)
    metrics_result['Harmonic coherence']['Generated chord coherence'] = np.round_(generated_chord_coherence, decimals=4)
    
    # loss, perplexity and accuracy of pitch model
    isPitch = True
    criterion = nn.CrossEntropyLoss(ignore_index=pitch_pad_idx)    
    testLoss_results_pitch, accuracy_results_pitch = mod.evaluate(modelPitch, pitch_to_ix, 
                                                                test_pitch_batched, test_duration_batched, 
                                                                test_chord_batched, test_next_chord_batched,
                                                                test_bass_batched, test_beat_batched, test_offset_batched,
                                                                criterion, con.BPTT, device, isPitch)
    
    # BLEU score
    bleu_pitch, bleu_duration = ev.BLEUscore(original_structuredSongs, generated_structuredSongs)
    
    metrics_result['Pitch']['Pitch_test-loss'] = np.round_(testLoss_results_pitch, decimals=4)
    metrics_result['Pitch']['Pitch_perplexity'] = np.round_(math.exp(testLoss_results_pitch), decimals=4)
    metrics_result['Pitch']['Pitch_accuracy'] = np.round_(accuracy_results_pitch * 100, decimals=4)
    metrics_result['Pitch']['Pitch_BLEU'] = np.round_(bleu_pitch, decimals=4)
    
    # loss, perplexity and accuracy of duration model
    isPitch = False
    criterion = nn.CrossEntropyLoss(ignore_index=duration_pad_idx)
    testLoss_results_duration, accuracy_results_duration = mod.evaluate(modelDuration, duration_to_ix, 
                                                                    test_pitch_batched, test_duration_batched, 
                                                                    test_chord_batched, test_next_chord_batched,
                                                                    test_bass_batched, test_beat_batched, test_offset_batched,
                                                                    criterion, con.BPTT, device, isPitch)
    
    metrics_result['Duration']['Duration_test-loss'] = np.round_(testLoss_results_duration, decimals=4)
    metrics_result['Duration']['Duration_perplexity'] = np.round_(math.exp(testLoss_results_duration), decimals=4)
    metrics_result['Duration']['Duration_accuracy'] = np.round_(accuracy_results_duration * 100, decimals=4)
    metrics_result['Duration']['Duration_BLEU'] = np.round_(bleu_duration, decimals=4)
    
    
    # Convert metrics dict to JSON and SAVE IT
    with open(path + 'metrics' + model_id + '.json', 'w') as fp:
        json.dump(metrics_result, fp, indent=4)
    

