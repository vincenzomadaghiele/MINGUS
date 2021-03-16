#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 08:56:16 2021

@author: vincenzomadaghiele
"""


import pretty_midi
import music21 as m21
import torch
import torch.nn as nn
import numpy as np
import json
import math
import time
import MINGUS_dataset_funct as dataset
import MINGUS_model as mod

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

# Constants
TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
BPTT = 128 # length of one note sequence (use powers of 2 for even divisions)
AUGMENTATION = False


def batchify(data, bsz, dict_to_ix, device):
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
    
    padded_num = [[dict_to_ix[x] for x in ex] for ex in data]
    
    data = torch.tensor(padded_num, dtype=torch.long)
    data = data.contiguous()
                
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    
    return data.to(device)


if __name__ == '__main__':
    
    # DATA LOADING
    print('Loading data...')
    songs_path = 'data/WjazzDB.json'
    with open(songs_path) as f:
        songs = json.load(f)
    
    pitch_train = []
    pitch_validation = []
    pitch_test = []
    
    duration_train = []
    duration_validation = []
    duration_test = []
    
    chord_train = []
    chord_validation = []
    chord_test = []
    
    bass_train = []
    bass_validation = []
    bass_test = []
    
    beat_train = []
    beat_validation = []
    beat_test = []
    
    
    for song in songs['train']:
        pitch_train.append(song['pitch'])
        duration_train.append(song['duration'])
        chord_train.append(song['chords'])
        bass_train.append(song['bass pitch'])
        beat_train.append(song['beats'])
            
    for song in songs['validation']:
        pitch_validation.append(song['pitch'])
        duration_validation.append(song['duration'])
        chord_validation.append(song['chords'])
        bass_validation.append(song['bass pitch'])
        beat_validation.append(song['beats'])
            
    for song in songs['test']:
        pitch_test.append(song['pitch'])
        duration_test.append(song['duration'])
        chord_test.append(song['chords'])
        bass_test.append(song['bass pitch'])
        beat_test.append(song['beats'])
    
    
    #%% COMPUTE VOCABS
    
    # use all pitch values 0-127
    vocabPitch = {} 
    for i in range(0,128):
        vocabPitch[i] = i
    vocabPitch[128] = 'R'
    # inverse dictionary
    pitch_to_ix = {v: k for k, v in vocabPitch.items()} 
    
    # use 12 duration values
    duration_to_ix = {} 
    duration_to_ix['full'] = 0
    duration_to_ix['half'] = 1
    duration_to_ix['quarter'] = 2
    duration_to_ix['8th'] = 3
    duration_to_ix['16th'] = 4
    duration_to_ix['dot half'] = 5
    duration_to_ix['dot quarter'] = 6
    duration_to_ix['dot 8th'] = 7
    duration_to_ix['dot 16th'] = 8
    duration_to_ix['half note triplet'] = 9
    duration_to_ix['quarter note triplet'] = 10
    duration_to_ix['8th note triplet'] = 11
    # inverse dictionary
    vocabDuration = {v: k for k, v in duration_to_ix.items()}
    
    # Beat dictionary
    vocabBeat = {} 
    for i in range(0,4):
        vocabBeat[i] = i
    # inverse dictionary
    beat_to_ix = {v: k for k, v in vocabBeat.items()} 
    
    
    #%% PRE-PROCESSING
    
    print('Pre-processing...')
    # Convert lists to array 

    pitch_train = np.array(pitch_train)
    pitch_validation = np.array(pitch_validation)
    pitch_test = np.array(pitch_test)
    
    duration_train = np.array(duration_train)
    duration_validation = np.array(duration_validation)
    duration_test = np.array(duration_test)
    
    bass_train = np.array(bass_train)
    bass_validation = np.array(bass_validation)
    bass_test = np.array(bass_test)
    
    beat_train = np.array(beat_train)
    beat_validation = np.array(beat_validation)
    beat_test = np.array(beat_test)
    
    
    # BUILD CHORD DICTIONARY
    
    # select unique chords
    unique_chords = []
    for chord_list in chord_train:
        for chord in chord_list:
            if chord not in unique_chords:
                unique_chords.append(chord)
    
    for chord_list in chord_validation:
        for chord in chord_list:
            if chord not in unique_chords:
                unique_chords.append(chord)
    
    for chord_list in chord_test:
        for chord in chord_list:
            if chord not in unique_chords:
                unique_chords.append(chord)
    
    # substitute for chord representation compatibility between music21 and WjazzDB
    new_unique_chords = []
    WjazzToMusic21 = {}
    WjazzToChordComposition = {}
    for chord in unique_chords:
        chord_components = [char for char in chord]
        for i in range(len(chord_components)):
            # substitute '-' with 'm'
            if chord_components[i] == '-':
                chord_components[i] = 'm'
            # substitute 'j' with 'M'
            if chord_components[i] == 'j':
                chord_components[i] = 'M'
            # substitute 'b' with '-'
            if i == 1 and chord_components[i] == 'b':
                chord_components[i] = '-'
            # change each 9# or 11# or 13# to #9 or #11 or #13
            if (chord_components[i] == '#' or chord_components[i] == 'b') and (chord_components[i-1] == '9' or (chord_components[i-1] == '1' and chord_components[i-2] == '1') or (chord_components[i-1] == '3' and chord_components[i-2] == '1')):
                if chord_components[i-1] == '9':
                    alteration = chord_components[i]
                    degree = chord_components[i-1]
                    chord_components[i-1] = alteration
                    chord_components[i] = degree
                else:
                    alteration = chord_components[i]
                    dx = chord_components[i-2]
                    ux = chord_components[i-1]
                    chord_components[i-2] = alteration
                    chord_components[i-1] = dx
                    chord_components[i] = ux
        # substitute 'alt' with '5#'
        if len(chord) > 3:
            if chord[-3:] == 'alt':
                chord_components[1] = '+'
                chord_components[2] = '7'
                del chord_components[3:]
            if chord[-4:] == 'sus7':
                chord_components[-4] = '7'
                chord_components[-3:] = 'sus'
            if chord[-5:] == 'sus79':
                chord_components[-5] = '7'
                chord_components[-4:] = 'sus'
            if chord[-3:] == 'j79':
                chord_components[-3:] = 'M7'
            if chord[-6:] == 'j7911#':
                chord_components[-6:] = 'M7#11'
        
        # reassemble chord
        new_chord_name = ''
        for component in chord_components:
            new_chord_name += component
            
        # special cases
        if chord == 'C-/Bb':
            new_chord_name = 'Cm/B-'
        if chord == 'Eb/Bb':
            new_chord_name = 'E-/B-' # MAKE THIS GENERAL 
        if chord == 'C-69':
            new_chord_name = 'Cm9'
        if chord == 'Gbj79':
            new_chord_name = 'G-M7'

        
        new_unique_chords.append(new_chord_name)
        
        print(chord)
        print(new_chord_name)
        
        if new_chord_name == 'NC':
            pitchNames = []
        else:
            h = m21.harmony.ChordSymbol(new_chord_name)
            pitchNames = [str(p) for p in h.pitches]
        
        print('%-10s%s' % (new_chord_name, '[' + (', '.join(pitchNames)) + ']'))
        
        # Update dictionaries
        WjazzToMusic21[chord] = new_chord_name
        WjazzToChordComposition[chord] = pitchNames
    
    
    #%%
    
    chord = 'G7913#'
    chord_components = [char for char in chord]
    if (chord_components[i] == '#' or chord_components[i] == 'b') and (chord_components[i-1] == '9' or (chord_components[i-1] == '1' and chord_components[i-2] == '1') or (chord_components[i-1] == '3' and chord_components[i-2] == '1')):
        if chord_components[i-1] == '9':
            alteration = chord_components[i]
            degree = chord_components[i-1]
            chord_components[i-1] = alteration
            chord_components[i] = degree
        else:
            alteration = chord_components[i]
            dx = chord_components[i-2]
            ux = chord_components[i-1]
            chord_components[i-2] = alteration
            chord_components[i-1] = dx
            chord_components[i] = ux
            
        
    #%%
    h = m21.harmony.ChordSymbol('G/Eb')
    pitchNames = [str(p) for p in h.pitches]
    print('%-10s%s' % ('G/Eb', '[' + (', '.join(pitchNames)) + ']'))
    
    
    #%% Data augmentation
    
    if AUGMENTATION:
        # augmentation is done by transposing the velocities and the pitch
        # velocity + 1 --> pitch + 1
        
        print('Data augmentation...')
        # ex. if augmentation_const = 4 the song will be transposed +4 and -4 times
        augmentation_const = 4 
        
        # training augmentation
        new_pitch = []
        new_duration = []
        for aug in range (1,augmentation_const):
            for i in range(pitch_train.shape[0]):
                new_pitch.append(pitch_train[i,:] + aug)
                new_duration.append(duration_train[i,:])
                new_pitch.append(pitch_train[i,:] - aug)
                new_duration.append(duration_train[i,:])
        
        pitch_train = np.array(new_pitch)
        duration_train = np.array(new_duration)
        
        # validation augmentation
        new_pitch = []
        new_duration = []
        for aug in range (1,augmentation_const):
            for i in range(pitch_validation.shape[0]):
                new_pitch.append(pitch_validation[i,:] + aug)
                new_duration.append(duration_validation[i,:])
                new_pitch.append(pitch_validation[i,:] - aug)
                new_duration.append(duration_validation[i,:])
        
        pitch_validation = np.array(new_pitch)
        duration_validation = np.array(new_duration)
        
        # test augmentation
        new_pitch = []
        new_duration = []
        for aug in range (1,augmentation_const):
            for i in range(pitch_test.shape[0]):
                new_pitch.append(pitch_test[i,:] + aug)
                new_duration.append(duration_test[i,:])
                new_pitch.append(pitch_test[i,:] - aug)
                new_duration.append(duration_test[i,:])
        
        pitch_test = np.array(new_pitch)
        duration_test = np.array(new_duration)
        
        # check for out of threshold pitch
        pitch_train[pitch_train > 127] = 127
        pitch_train[pitch_train < 0] = 0
        pitch_validation[pitch_validation > 127] = 127
        pitch_validation[pitch_validation < 0] = 0
        pitch_test[pitch_test > 127] = 127
        pitch_test[pitch_test < 0] = 0
    
    '''
    #%% Reshape according to sequence length
    
    print('Reshaping...')
    
    new_pitch = []
    new_duration = []
    for i in range(pitch_train.shape[0]):
        for j in range(int(pitch_train.shape[1]/BPTT)):
            new_pitch.append(pitch_train[i, j * BPTT:(j+1) * BPTT])
            new_duration.append(duration_train[i, j * BPTT:(j+1) * BPTT])

    pitch_train = np.array(new_pitch)
    duration_train = np.array(new_duration)
    
    # reshape validation
    new_pitch = []
    new_duration = []
    for i in range(pitch_validation.shape[0]):
        for j in range(int(pitch_validation.shape[1]/BPTT)):
            new_pitch.append(pitch_validation[i, j * BPTT:(j+1) * BPTT])
            new_duration.append(duration_validation[i, j * BPTT:(j+1) * BPTT])

    pitch_validation = np.array(new_pitch)
    duration_validation = np.array(new_duration)
    
    # reshape test
    new_pitch = []
    new_duration = []
    for i in range(pitch_test.shape[0]):
        for j in range(int(pitch_test.shape[1]/BPTT)):
            new_pitch.append(pitch_test[i, j * BPTT:(j+1) * BPTT])
            new_duration.append(duration_test[i, j * BPTT:(j+1) * BPTT])

    pitch_test = np.array(new_pitch)
    duration_test = np.array(new_duration)
    
    
    #%% TOKENIZE AND BATCHIFY
    
    
    pitch_train = pitch_train[0:1000,:]
    pitch_validation = pitch_validation[0:1000,:]
    pitch_test = pitch_test[0:1000,:]
    duration_train = duration_train[0:1000,:]
    duration_validation = duration_validation[0:1000,:]
    duration_test = duration_test[0:1000,:]
    
    
    print('Batching...')
    
    train_pitch_batched = batchify(pitch_train, TRAIN_BATCH_SIZE, pitch_to_ix, device)
    val_pitch_batched = batchify(pitch_validation, EVAL_BATCH_SIZE, pitch_to_ix, device)
    test_pitch_batched = batchify(pitch_test, EVAL_BATCH_SIZE, pitch_to_ix, device)
    
    train_duration_batched = batchify(duration_train, TRAIN_BATCH_SIZE, duration_to_ix, device)
    val_duration_batched = batchify(duration_validation, EVAL_BATCH_SIZE, duration_to_ix, device)
    test_duration_batched = batchify(duration_test, EVAL_BATCH_SIZE, duration_to_ix, device)    
    '''
