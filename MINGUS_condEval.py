#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:15:44 2021

@author: vincenzomadaghiele
"""

import json
import numpy as np
import math
import torch
import torch.nn as nn
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con
import MINGUS_eval_funct as ev
from nltk.translate.bleu_score import corpus_bleu

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':

    # LOAD DATA
    
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
    
    # Import model
    savePATHpitch = 'models/MINGUSpitch_10epochs_seqLen35_NottinghamDB.pt'
    modelPitch.load_state_dict(torch.load(savePATHpitch, map_location=torch.device('cpu')))
    
    
    # DURATION MODEL
    isPitch = False
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
    
    # Import model
    savePATHduration = 'models/MINGUSduration_10epochs_seqLen35_NottinghamDB.pt'
    modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))

    
    #%% BLEU score
    
    gen_common_path = 'output/gen4eval_NottinghamDB/'
    original_subpath = 'original/'
    generated_subpath = 'generated/'
    
    path = gen_common_path + original_subpath + 'NottinghamDB_original.json'
    with open(path) as f:
        original_structuredSongs = json.load(f)
        
    path = gen_common_path + generated_subpath + 'NottinghamDB_generated.json'
    with open(path) as f:
        generated_structuredSongs = json.load(f)
    
    original_tunes_pitch = []
    original_tunes_duration = []
    for tune in original_structuredSongs:
        newtune_pitch = []
        newtune_duration = []
        for bar in tune['bars']:
            for beat in bar['beats']:
                for pitch in beat['pitch']:
                    newtune_pitch.append(pitch)
                for duration in beat['duration']:
                    newtune_duration.append(duration)
        original_tunes_pitch.append(newtune_pitch)
        original_tunes_duration.append(newtune_duration)
    
    generated_tunes_pitch = []
    generated_tunes_duration = []
    for tune in original_structuredSongs:
        newtune_pitch = []
        newtune_duration = []
        for bar in tune['bars']:
            for beat in bar['beats']:
                for pitch in beat['pitch']:
                    newtune_pitch.append(pitch)
                for duration in beat['duration']:
                    newtune_duration.append(duration)
        generated_tunes_pitch.append(newtune_pitch)
        generated_tunes_duration.append(newtune_duration)
    
    num_seq = len(generated_tunes_pitch) # number of generated sequences
    num_ref = 4 # number of reference examples for each generated sequence
    
    reference_pitch = [original_tunes_pitch[i:i+num_ref-1] for i in range(0,num_ref*num_seq,num_ref)]
    reference_duration = [original_tunes_duration[i:i+num_ref-1] for i in range(0,num_ref*num_seq,num_ref)]
    
    bleu_pitch = corpus_bleu(reference_pitch[:4], generated_tunes_pitch[:4])
    bleu_duration = corpus_bleu(reference_duration[:4], generated_tunes_duration[:4])

    
    #%% METRICS DICTIONARY
    
    # Instanciate dictionary
    metrics_result = {}
    metrics_result['MGEval'] = {}
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
    original_path = 'output/gen4eval_NottinghamDB/original/*.mid'
    generated_path = 'output/gen4eval_NottinghamDB/generated/*.mid'
    MGEresults = ev.MGEval(original_path, generated_path, num_of_generations)
    metrics_result['MGEval'] = MGEresults
    
    # loss, perplexity and accuracy of pitch model
    isPitch = True
    criterion = nn.CrossEntropyLoss(ignore_index=pitch_pad_idx)    
    testLoss_results_pitch, accuracy_results_pitch = mod.evaluate(modelPitch, pitch_to_ix, 
                                                                test_pitch_batched, test_duration_batched, test_chord_batched,
                                                                test_bass_batched, test_beat_batched,
                                                                criterion, con.BPTT, device, isPitch)
    
    metrics_result['Pitch']['Pitch_test-loss'] = np.round_(testLoss_results_pitch, decimals=4)
    metrics_result['Pitch']['Pitch_perplexity'] = np.round_(math.exp(testLoss_results_pitch), decimals=4)
    metrics_result['Pitch']['Pitch_accuracy'] = np.round_(accuracy_results_pitch * 100, decimals=4)
    metrics_result['Pitch']['Pitch_BLEU'] = np.round_(bleu_pitch, decimals=4)
    
    # loss, perplexity and accuracy of duration model
    isPitch = False
    criterion = nn.CrossEntropyLoss(ignore_index=duration_pad_idx)
    testLoss_results_duration, accuracy_results_duration = mod.evaluate(modelDuration, duration_to_ix, 
                                                                    test_pitch_batched, test_duration_batched, test_chord_batched,
                                                                    test_bass_batched, test_beat_batched,
                                                                    criterion, con.BPTT, device, isPitch)
    
    metrics_result['Duration']['Duration_test-loss'] = np.round_(testLoss_results_duration, decimals=4)
    metrics_result['Duration']['Duration_perplexity'] = np.round_(math.exp(testLoss_results_duration), decimals=4)
    metrics_result['Duration']['Duration_accuracy'] = np.round_(accuracy_results_duration * 100, decimals=4)
    metrics_result['Duration']['Duration_BLEU'] = np.round_(bleu_duration, decimals=4)
    
    # Convert metrics dict to JSON and SAVE IT    
    with open('metrics/metrics_result.json', 'w') as fp:
        json.dump(metrics_result, fp, indent=4)
    

