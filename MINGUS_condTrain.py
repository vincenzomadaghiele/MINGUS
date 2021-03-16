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
SEGMENTATION = True


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
    #init_token = '<sos>'
    #eos_token = '<eos>'
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
    
    if SEGMENTATION:
        padded = pad(data)
        padded_num = [[dict_to_ix[x] for x in ex] for ex in padded]
    else:
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

def separateSeqs(seq_pitch, seq_duration,
                 seq_chord, seq_bass, seq_beat,
                 segment_length = 35):
    # Separate the songs into single melodies in order to avoid 
    # full batches of pad tokens
        
    tot_pitch = []
    tot_duration = []
    tot_chord = []
    tot_bass = []
    tot_beat = []

    new_pitch = []
    new_duration = []
    new_chord = []
    new_bass = []
    new_beat = []
    
    long_dur = ['full', 'half', 'quarter', 'dot half', 'dot quarter', 
                    'dot 8th', 'half note triplet', 'quarter note triplet']
    
    #long_dur = ['full', 'half', 'quarter', 'dot half', 'dot quarter']
    
    counter = 0
    for i in range(len(seq_pitch)):
        
        new_pitch.append(seq_pitch[i])
        new_duration.append(seq_duration[i])
        new_chord.append(seq_chord[i])
        new_bass.append(seq_bass[i])
        new_beat.append(seq_beat[i])
        
        counter += 1
        if seq_pitch[i] == 'R' and seq_duration[i] in long_dur:
            if counter > int(segment_length/3):
                tot_pitch.append(np.array(new_pitch, dtype=object))
                tot_duration.append(np.array(new_duration, dtype=object))
                tot_chord.append(np.array(new_chord, dtype=object))
                tot_bass.append(np.array(new_bass, dtype=object))
                tot_beat.append(np.array(new_beat, dtype=object))
                new_pitch = []
                new_duration = []
                new_chord = []
                new_bass = []
                new_beat = []
                counter = 0
        elif counter == segment_length:
            tot_pitch.append(np.array(new_pitch, dtype=object))
            tot_duration.append(np.array(new_duration, dtype=object))
            tot_chord.append(np.array(new_chord, dtype=object))
            tot_bass.append(np.array(new_bass, dtype=object))
            tot_beat.append(np.array(new_beat, dtype=object))
            new_pitch = []
            new_duration = []
            new_chord = []
            new_bass = []
            new_beat = []
            counter = 0
            
    return tot_pitch, tot_duration, tot_chord, tot_bass, tot_beat
    
def segmentDataset(pitch_data, duration_data, 
                   chord_data, bass_data, beat_data,
                   segment_length = 35):
    
    pitch_segmented = []
    duration_segmented = []
    chord_segmented = []
    bass_segmented = []
    beat_segmented = []
    for i in range(len(pitch_data)):
        pitch_sep, duration_sep, chord_sep, bass_sep, beat_sep = separateSeqs(pitch_data[i], 
                                                                              duration_data[i], 
                                                                              chord_data[i],
                                                                              bass_data[i],
                                                                              beat_data[i], 
                                                                              segment_length)
        for seq in pitch_sep:
            pitch_segmented.append(seq)
        for seq in duration_sep:
            duration_segmented.append(seq)
        for seq in chord_sep:
            chord_segmented.append(seq)
        for seq in bass_sep:
            bass_segmented.append(seq)
        for seq in beat_sep:
            beat_segmented.append(seq)
            
    pitch_segmented = np.array(pitch_segmented, dtype=object)
    duration_segmented = np.array(duration_segmented, dtype=object)
    chord_segmented = np.array(duration_segmented, dtype=object)
    bass_segmented = np.array(duration_segmented, dtype=object)
    beat_segmented = np.array(duration_segmented, dtype=object)
        
    return pitch_segmented, duration_segmented, chord_segmented, bass_segmented, beat_segmented


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
    
    print('Computing vocabs...')
    
    # use all pitch values 0-127
    # vocabPitch is used to encode also bass and chord values
    vocabPitch = {} 
    for i in range(0,128):
        vocabPitch[i] = i
    vocabPitch[128] = 'R'
    vocabPitch[129] = '<pad>'
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
    duration_to_ix['pad'] = 12
    # inverse dictionary
    vocabDuration = {v: k for k, v in duration_to_ix.items()}
    
    # Beat dictionary
    vocabBeat = {} 
    for i in range(0,4):
        vocabBeat[i] = i
    vocabBeat[5] = '<pad>'
    # inverse dictionary
    beat_to_ix = {v: k for k, v in vocabBeat.items()}     
    
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
    # to do this I have extracted all unique chords and operated some simplification
    # to the chord notes in order for them to be recognized by music21
    new_unique_chords = []
    WjazzChords = []
    WjazzToMusic21 = {}
    WjazzToChordComposition = {}
    WjazzToMidiChords = {}
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
            if chord_components[i] == 'b' and chord_components[i-2] == '/':
                chord_components[i] = '-'
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
            if chord[-3:] == '-69':
                chord_components[-3:] = 'm9'
            if chord[-3:] == '-79':
                chord_components[-3:] = 'm7'
            if chord[-5:] == '-7911':
                chord_components[-5:] = 'm7'
            if chord[-7:] == 'sus7913':
                chord_components[-7:] = '7sus'
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
        if chord == 'Eb-79/Ab':
            new_chord_name = 'E-m7'
        if chord == 'Aj7911#/Ab':
            new_chord_name = 'AM7#11/A-'
        if chord == 'Db-69/Ab':
            new_chord_name = 'D-m9/A-'
        if chord == 'Db-69/Db':
            new_chord_name = 'D-m9/D-'
        if chord == 'Dbj7911#/C':
            new_chord_name = 'D-M7#11/C'
        if chord == 'C-j7913':
            new_chord_name = 'CmM7'
        if chord == 'Bb-7913':
            new_chord_name = 'B-m7'

        if new_chord_name == 'NC':
            pitchNames = []
        else:
            h = m21.harmony.ChordSymbol(new_chord_name)
            pitchNames = [str(p) for p in h.pitches]
        
        #print('%-10s%s' % (new_chord_name, '[' + (', '.join(pitchNames)) + ']'))
        
        midiChord = []
        multi_hot = np.zeros(12)
        for p in pitchNames:
            
            # midi conversion
            c = m21.pitch.Pitch(p)
            midiChord.append(c.midi)
            
            # multi-hot
            # C C#/D- D D#/E- E E#/F F#/G- G G#/A- A A#/B- B 
            # 0   1   2   3   4   5    6   7   8   9  10   11
            if p[0] == 'C' and p[1] == '-':
                multi_hot[11] = 1
            elif p[0] == 'C' and p[1] == '#':
                multi_hot[1] = 1
            elif p[0] == 'C' :
                multi_hot[0] = 1
            
            elif p[0] == 'D' and p[1] == '-':
                multi_hot[1] = 1
            elif p[0] == 'D' and p[1] == '#':
                multi_hot[3] = 1
            elif p[0] == 'D' :
                multi_hot[2] = 1
            
            elif p[0] == 'E' and p[1] == '-':
                multi_hot[3] = 1
            elif p[0] == 'E' and p[1] == '#':
                multi_hot[5] = 1
            elif p[0] == 'E' :
                multi_hot[4] = 1
            
            elif p[0] == 'F' and p[1] == '-':
                multi_hot[4] = 1
            elif p[0] == 'F' and p[1] == '#':
                multi_hot[6] = 1
            elif p[0] == 'F' :
                multi_hot[7] = 1
            
            elif p[0] == 'G' and p[1] == '-':
                multi_hot[6] = 1
            elif p[0] == 'G' and p[1] == '#':
                multi_hot[8] = 1
            elif p[0] == 'G' :
                multi_hot[7] = 1
            
            elif p[0] == 'A' and p[1] == '-':
                multi_hot[8] = 1
            elif p[0] == 'A' and p[1] == '#':
                multi_hot[10] = 1
            elif p[0] == 'A' :
                multi_hot[9] = 1
            
            elif p[0] == 'B' and p[1] == '-':
                multi_hot[10] = 1
            elif p[0] == 'B' and p[1] == '#':
                multi_hot[0] = 1
            elif p[0] == 'B' :
                multi_hot[11] = 1
        

        # Update dictionaries
        new_unique_chords.append(new_chord_name)
        WjazzToMusic21[chord] = new_chord_name
        WjazzToChordComposition[chord] = pitchNames
        WjazzToMidiChords[chord] = midiChord
        
        NewChord = {}
        NewChord['Wjazz name'] = chord
        NewChord['music21 name'] = new_chord_name
        NewChord['chord composition'] = pitchNames
        NewChord['midi chord composition'] = midiChord
        NewChord['one-hot encoding'] = multi_hot.tolist()
        WjazzChords.append(NewChord)
    
    
    #%% PRE-PROCESSING
    
    print('Pre-processing...')
    # Convert lists to array 

    pitch_train = np.array(pitch_train)
    pitch_validation = np.array(pitch_validation)
    pitch_test = np.array(pitch_test)
    
    duration_train = np.array(duration_train)
    duration_validation = np.array(duration_validation)
    duration_test = np.array(duration_test)
   
    new_chord_train = []
    for song in chord_train:
        new_song_chords = []
        for chord in song:
            new_song_chords.append(WjazzToMidiChords[chord])
        new_chord_train.append(new_song_chords)
    
    new_chord_validation = []
    for song in chord_validation:
        new_song_chords = []
        for chord in song:
            new_song_chords.append(WjazzToMidiChords[chord])
        new_chord_validation.append(new_song_chords)
    
    new_chord_test = []
    for song in chord_test:
        new_song_chords = []
        for chord in song:
            new_song_chords.append(WjazzToMidiChords[chord])
        new_chord_test.append(new_song_chords)
    
    chord_train = np.array(new_chord_train)
    chord_validation = np.array(new_chord_validation)
    chord_test = np.array(new_chord_test)
    
    bass_train = np.array(bass_train)
    bass_validation = np.array(bass_validation)
    bass_test = np.array(bass_test)
    
    beat_train = np.array(beat_train)
    beat_validation = np.array(beat_validation)
    beat_test = np.array(beat_test)
    
    
    
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
        new_chord = []
        new_bass = []
        new_beat = []
        for aug in range (1,augmentation_const):
            for i in range(pitch_train.shape[0]):
                new_pitch.append(pitch_train[i] + aug) # SOLVE: CANNOT SUM BECAUSE OF RESTS
                new_duration.append(duration_train[i])
                new_chord.append(chord_train[i] + aug)
                new_bass.append(bass_train[i] + aug)
                new_beat.append(beat_train[i])
                
                new_pitch.append(pitch_train[i] - aug)
                new_duration.append(duration_train[i])
                new_chord.append(chord_train[i] - aug)
                new_bass.append(bass_train[i] - aug)
                new_beat.append(beat_train[i])
        
        pitch_train = np.array(new_pitch)
        duration_train = np.array(new_duration)
        chord_train = np.array(new_chord)
        bass_train = np.array(new_bass)
        beat_train = np.array(new_beat)
        
        # validation augmentation
        new_pitch = []
        new_duration = []
        new_chord = []
        new_bass = []
        new_beat = []
        for aug in range (1,augmentation_const):
            for i in range(pitch_validation.shape[0]):
                new_pitch.append(pitch_validation[i] + aug)
                new_duration.append(duration_validation[i])
                new_chord.append(chord_validation[i] + aug)
                new_bass.append(bass_validation[i] + aug)
                new_beat.append(beat_validation[i])
                
                new_pitch.append(pitch_validation[i] - aug)
                new_duration.append(duration_validation[i])
                new_chord.append(chord_validation[i] - aug)
                new_bass.append(bass_validation[i] - aug)
                new_beat.append(beat_validation[i])
        
        pitch_validation = np.array(new_pitch)
        duration_validation = np.array(new_duration)
        chord_validation = np.array(new_chord)
        bass_validation = np.array(new_bass)
        beat_validation = np.array(new_beat)
        
        # test augmentation
        new_pitch = []
        new_duration = []
        new_chord = []
        new_bass = []
        new_beat = []
        for aug in range (1,augmentation_const):
            for i in range(pitch_test.shape[0]):
                new_pitch.append(pitch_test[i] + aug)
                new_duration.append(duration_test[i])
                new_chord.append(chord_test[i] + aug)
                new_bass.append(bass_test[i] + aug)
                new_beat.append(beat_test[i])
                
                new_pitch.append(pitch_test[i] - aug)
                new_duration.append(duration_test[i])
                new_chord.append(chord_test[i] - aug)
                new_bass.append(bass_test[i] - aug)
                new_beat.append(beat_test[i])
        
        pitch_test = np.array(new_pitch)
        duration_test = np.array(new_duration)
        chord_test = np.array(new_chord)
        bass_test = np.array(new_bass)
        beat_test = np.array(new_beat)
        
        
        # check for out of threshold pitch
        pitch_train[pitch_train > 127] = 127
        pitch_train[pitch_train < 0] = 0
        pitch_validation[pitch_validation > 127] = 127
        pitch_validation[pitch_validation < 0] = 0
        pitch_test[pitch_test > 127] = 127
        pitch_test[pitch_test < 0] = 0
        
        # probably not necessary with chords
        
        bass_train[bass_train > 127] = 127
        bass_train[bass_train < 0] = 0
        bass_validation[bass_validation > 127] = 127
        bass_validation[bass_validation < 0] = 0
        bass_test[bass_test > 127] = 127
        bass_test[bass_test < 0] = 0
    
    
    #%% Reshape according to sequence length
    # At the end of this step the datasets should be composed 
    # of equal length melodies
    # SEGMENTATION --> padded segmented melodies
    # !SEGMENTATION --> equal length not padded note sequences
    
    if SEGMENTATION:
        print('Melody segmentation...')
        
        pitch_train, duration_train, chord_train, bass_train, beat_train = segmentDataset(pitch_train, 
                                                                                          duration_train, 
                                                                                          chord_train,
                                                                                          bass_train,
                                                                                          beat_train,
                                                                                          BPTT)
        pitch_validation, duration_validation, chord_validation, bass_validation, beat_validation = segmentDataset(pitch_validation, 
                                                                                                                   duration_validation, 
                                                                                                                   chord_validation,
                                                                                                                   bass_validation,
                                                                                                                   beat_validation,
                                                                                                                   BPTT)
        pitch_test, duration_test, chord_test, bass_test, beat_test = segmentDataset(pitch_test, 
                                                                                     duration_test, 
                                                                                     chord_test,
                                                                                     bass_test,
                                                                                     beat_test,
                                                                                     BPTT)

    else:
        print('Reshaping...')
        
        new_pitch = []
        new_duration = []
        new_chord = []
        new_bass = []
        new_beat = []
        for i in range(pitch_train.shape[0]):
            for j in range(int(len(pitch_train[i])/BPTT)):
                new_pitch.append(pitch_train[j * BPTT:(j+1) * BPTT])
                new_duration.append(duration_train[j * BPTT:(j+1) * BPTT])
                new_chord.append(chord_train[j * BPTT:(j+1) * BPTT])
                new_bass.append(bass_train[j * BPTT:(j+1) * BPTT])
                new_beat.append(beat_train[j * BPTT:(j+1) * BPTT])
    
        pitch_train = np.array(new_pitch)
        duration_train = np.array(new_duration)
        chord_train = np.array(new_chord)
        bass_train = np.array(new_bass)
        beat_train = np.array(new_beat)
        
        # reshape validation
        new_pitch = []
        new_duration = []
        new_chord = []
        new_bass = []
        new_beat = []
        for i in range(pitch_validation.shape[0]):
            for j in range(int(len(pitch_validation[i])/BPTT)):
                new_pitch.append(pitch_validation[j * BPTT:(j+1) * BPTT])
                new_duration.append(duration_validation[j * BPTT:(j+1) * BPTT])
                new_chord.append(chord_validation[j * BPTT:(j+1) * BPTT])
                new_bass.append(bass_validation[j * BPTT:(j+1) * BPTT])
                new_beat.append(beat_validation[j * BPTT:(j+1) * BPTT])
    
        pitch_validation = np.array(new_pitch)
        duration_validation = np.array(new_duration)
        chord_validation = np.array(new_chord)
        bass_validation = np.array(new_bass)
        beat_validation = np.array(new_beat)
        
        # reshape test
        new_pitch = []
        new_duration = []
        new_chord = []
        new_bass = []
        new_beat = []
        for i in range(pitch_test.shape[0]):
            for j in range(int(len(pitch_test[i])/BPTT)):
                new_pitch.append(pitch_test[j * BPTT:(j+1) * BPTT])
                new_duration.append(duration_test[j * BPTT:(j+1) * BPTT])
                new_chord.append(chord_test[j * BPTT:(j+1) * BPTT])
                new_bass.append(bass_test[j * BPTT:(j+1) * BPTT])
                new_beat.append(beat_test[j * BPTT:(j+1) * BPTT])
    
        pitch_test = np.array(new_pitch)
        duration_test = np.array(new_duration)
        chord_test = np.array(new_chord)
        bass_test = np.array(new_bass)
        beat_test = np.array(new_beat)
    
    
    
    #%% TOKENIZE AND BATCHIFY
    
    '''
    pitch_train = pitch_train[0:1000,:]
    pitch_validation = pitch_validation[0:1000,:]
    pitch_test = pitch_test[0:1000,:]
    duration_train = duration_train[0:1000,:]
    duration_validation = duration_validation[0:1000,:]
    duration_test = duration_test[0:1000,:]
    '''
    
    print('Batching...')
    
    train_pitch_batched = batchify(pitch_train, TRAIN_BATCH_SIZE, pitch_to_ix, device)
    val_pitch_batched = batchify(pitch_validation, EVAL_BATCH_SIZE, pitch_to_ix, device)
    test_pitch_batched = batchify(pitch_test, EVAL_BATCH_SIZE, pitch_to_ix, device)
    
    train_duration_batched = batchify(duration_train, TRAIN_BATCH_SIZE, duration_to_ix, device)
    val_duration_batched = batchify(duration_validation, EVAL_BATCH_SIZE, duration_to_ix, device)
    test_duration_batched = batchify(duration_test, EVAL_BATCH_SIZE, duration_to_ix, device)    
    
