#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 09:01:54 2021

@author: vincenzomadaghiele
"""

import pretty_midi
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
                 con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
    
    #train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
    #val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getValidationData()
    #test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTestData()
    
    songs = WjazzDB.getOriginalSongDict()
    vocabPitch, vocabDuration, vocabBeat = WjazzDB.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix = WjazzDB.getInverseVocabs()
    WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = WjazzDB.getChordDicts()


    #%% Analysis
    
    '''
    pitch_tot = []
    duration_tot = []
    bass_tot = []
    for song in songs['train']:
        pitch = song['pitch']
        durations = song['duration']
        bass = song['bass pitch']
        pitch_tot.append(pitch)
        duration_tot.append(durations)
        bass_tot.append(bass)
        
    pitchlist = [item for items in pitch_tot for item in items]
    durationlist = [item for items in duration_tot for item in items]
    basslist = [item for items in bass_tot for item in items]
    
    import matplotlib.pyplot as plt
    # define window size, output and axes
    fig, ax = plt.subplots(figsize=[8,6])
    ax.set_title("pitch hist")
    ax.set_xlabel("pitch")
    ax.set_ylabel("frequency")
    N, bins, patches = ax.hist(pitchlist, bins=127, color="#777777") #initial color of all bins
    plt.show()
    
    # define window size, output and axes
    fig, ax = plt.subplots(figsize=[8,6])
    ax.set_title("duration hist")
    ax.set_xlabel("pitch")
    ax.set_ylabel("frequency")
    N, bins, patches = ax.hist(durationlist, bins=17, color="#777777") #initial color of all bins
    plt.show()
    
    # define window size, output and axes
    fig, ax = plt.subplots(figsize=[8,6])
    ax.set_title("bass hist")
    ax.set_xlabel("pitch")
    ax.set_ylabel("frequency")
    N, bins, patches = ax.hist(basslist, bins=127, color="#777777") #initial color of all bins
    plt.show()
    '''
    

    #%% Conversion to MIDI

    trialSong = songs['train'][2]
    notes = trialSong['pitch']
    durations = trialSong['duration']
    beats = trialSong['beats']
    bars = trialSong['bars']
    beat_duration_sec = trialSong['beat duration [sec]']
    bass = trialSong['bass pitch']
    chords = trialSong['chords']
    
    unit = beat_duration_sec * 4 / 192.
    # possible note durations in seconds 
    # (it is possible to add representations - include 32nds, quintuplets...):
    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
    possible_durations = [unit * 192, unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3, 
                                              unit * 144, unit * 72, unit * 36, unit * 18, unit * 9, 
                                              unit * 64, unit * 32, unit * 16, unit * 8, unit * 4, unit * 2]
    
    # Define durations dictionary
    dur_dict = {}
    dur_dict[possible_durations[0]] = 'full'
    dur_dict[possible_durations[1]] = 'half'
    dur_dict[possible_durations[2]] = 'quarter'
    dur_dict[possible_durations[3]] = '8th'
    dur_dict[possible_durations[4]] = '16th'
    dur_dict[possible_durations[5]] = '32th'
    dur_dict[possible_durations[6]] = '64th'
    dur_dict[possible_durations[7]] = 'dot half'
    dur_dict[possible_durations[8]] = 'dot quarter'
    dur_dict[possible_durations[9]] = 'dot 8th'
    dur_dict[possible_durations[10]] = 'dot 16th'
    dur_dict[possible_durations[11]] = 'dot 32th'
    dur_dict[possible_durations[12]] = 'half note triplet'
    dur_dict[possible_durations[13]] = 'quarter note triplet'
    dur_dict[possible_durations[14]] = '8th note triplet'
    dur_dict[possible_durations[15]] = '16th note triplet'
    dur_dict[possible_durations[16]] = '32th note triplet'
    dur_dict[possible_durations[17]] = '64th note triplet'
    
    inv_dur_dict = {v: k for k, v in dur_dict.items()}
    
    '''
    # Construct a PrettyMIDI object.
    pm = pretty_midi.PrettyMIDI(initial_tempo=trialSong['avgtempo'])
    # Add a piano instrument
    inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    pm.instruments.append(inst)
    velocity = 100
    offset = 0
    # for each note
    for i in range(min(len(notes),len(durations))):
        if notes[i] != '<pad>' and durations[i] != '<pad>':
            if notes[i] == 'R':
                duration = inv_dur_dict[durations[i]]
            else:
                pitch = int(notes[i])
                duration = inv_dur_dict[durations[i]]
                start = offset
                end = offset + duration
                inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
            offset += duration
    
    pm.write('output/equal.mid')
    '''
    
    #%% Construct a PrettyMIDI object.
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=trialSong['avgtempo'])
    # Add a piano instrument
    solo = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    chords_inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    bass_inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    pm.instruments.append(solo)
    pm.instruments.append(chords_inst)
    pm.instruments.append(bass_inst)
    velocity = 100
    offset = 0
    previous_bass = 0
    previous_chord = [0,0,0,0]
    #beats_bass_dur = 0
    #beats_chord_dur = 0

    # for each note
    for i in range(len(notes)-1):
        if notes[i] != '<pad>' and durations[i] != '<pad>':
            if notes[i] == 'R':
                duration = inv_dur_dict[durations[i]]
                start = offset
                end = offset + duration
                
                # SOLVE BEAT COUNT PROBLEM!
                # check for new bass note
                if int(bass[i]) != previous_bass:
                    # How many beats does the bass note last ? 
                    beats_bass_dur = beats[i+1]
                    for j in range(i,len(notes)-1):
                        if bass[j+1] != int(bass[i]):
                            break
                        if beats[j+1] != beats[j]:
                            if beats[j+1] != 4:
                                beats_bass_dur += beats[j+1] - beats[j]
                            else:
                                beats_bass_dur += beats[j+1]
                    
                    bass_duration = inv_dur_dict['quarter'] * beats_bass_dur
                    bass_start = offset
                    bass_end = offset + bass_duration
                    bass_inst.notes.append(pretty_midi.Note(velocity, int(bass[i]), bass_start, bass_end))
                    previous_bass = int(bass[i])
                
                # check for new chord
                if chords[i] != previous_chord:
                    # How many beats does the bass note last ? 
                    beats_chord_dur = beats[i+1]
                    for j in range(i,len(notes)-1):
                        if chords[j+1] != chords[i]:
                            break
                        if beats[j+1] != beats[j]:
                            if beats[j+1] != 1:
                                beats_chord_dur += beats[j+1] - beats[j]
                            else:
                                beats_chord_dur += beats[j+1]
                            
                    print('Chord: %s lasts %d beats' % (chords[i], beats_chord_dur))
                    
                    chord_duration = inv_dur_dict['quarter'] * beats_chord_dur
                    chord_start = offset
                    chord_end = offset + chord_duration
                    for chord_pitch in WjazzToMidiChords[chords[i]]:
                        chords_inst.notes.append(pretty_midi.Note(velocity, int(chord_pitch), start, end))
                        previous_chord = chords[i]      
                
                '''
                current_chord = chords[i]
                if current_chord != previous_chord:
                    for chord_pitch in WjazzToMidiChords[current_chord]:
                        chords_inst.notes.append(pretty_midi.Note(velocity, int(chord_pitch), start, end))
                        previous_chord = current_chord
                '''
            else:
                pitch = int(notes[i])
                duration = inv_dur_dict[durations[i]]
                start = offset
                end = offset + duration
                solo.notes.append(pretty_midi.Note(velocity, pitch, start, end))
                
                # check for new bass note
                if int(bass[i]) != previous_bass:
                    # How many beats does the bass note last ? 
                    beats_bass_dur = beats[i+1]
                    for j in range(i,len(notes)-1):
                        if bass[j+1] != int(bass[i]):
                            break
                        if beats[j+1] != beats[j]:
                            if beats[j+1] != 4:
                                beats_bass_dur += beats[j+1] - beats[j]
                            else:
                                beats_bass_dur += beats[j+1]
                    
                    bass_duration = inv_dur_dict['quarter'] * beats_bass_dur
                    bass_start = offset
                    bass_end = offset + bass_duration
                    bass_inst.notes.append(pretty_midi.Note(velocity, int(bass[i]), bass_start, bass_end))
                    previous_bass = int(bass[i])
                
                # check for new chord
                if chords[i] != previous_chord:
                    # How many beats does the bass note last ? 
                    beats_chord_dur = beats[i+1]
                    for j in range(i,len(notes)-1):
                        if chords[j+1] != chords[i]:
                            break
                        if beats[j+1] != beats[j]:
                            if beats[j+1] != 1:
                                beats_chord_dur += beats[j+1] - beats[j]
                            else:
                                beats_chord_dur += beats[j+1]
                    
                    print('Chord: %s lasts %d beats' % (chords[i], beats_chord_dur))
                    
                    chord_duration = inv_dur_dict['quarter'] * beats_chord_dur
                    chord_start = offset
                    chord_end = offset + chord_duration
                    for chord_pitch in WjazzToMidiChords[chords[i]]:
                        chords_inst.notes.append(pretty_midi.Note(velocity, int(chord_pitch), start, end))
                        previous_chord = chords[i] 
                
                '''
                current_chord = chords[i]
                if current_chord != previous_chord:
                    for chord_pitch in WjazzToMidiChords[current_chord]:
                        chords_inst.notes.append(pretty_midi.Note(velocity, int(chord_pitch), start, end))
                        previous_chord = current_chord
                '''
                
            offset += duration
    
    pm.write('output/equal.mid')
    
    
    #%% 
    
    offset = 0
    current_bass_pitch = bass[0]
    current_beat = beats[0]
    num_bass_beats = 0
    for i in range(len(bass)):
        current_bar = trialSong['bars']
        current_beat = trialSong['beats']
        #current_bass_pitch = bass[i]
        if bass[i] == current_bass_pitch and beats[i] != current_beat:
            num_bass_beats += current_beat - beats[i]
        elif bass[i] != current_bass_pitch and beats[i] != current_beat:
            duration = inv_dur_dict[durations[i]]
            start = offset
            end = offset + duration
            inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
        
        elif bass[i] != current_bass_pitch and beats[i] == current_beat:
            print('Error: bass change inside a beat')
    
    
    #%%
    
    num_bars = max(bars)
    new_bars = []
    new_beats = []
    new_bass = []
    new_chords = []
    
    for i in range(num_bars):
        for j in range(1,5):
            new_bars.append(i)
            new_beats.append(j)
    
    current_bass = bass[0]
    current_bar = bars[0]
    current_beat = beats[0]
    new_bass.append(current_bass)
    for i in range(len(bass)):
        bass[i]
        bars[i]
        beats[i]
        if beats[i] != current_beat:
            new_bass.append(bass[i])
            current_beat = beats[i]
    
    
    current_bar = bars[0]
    current_beat = beats[0]
    count_bars = 0
    count_beats = 0
    for i in range(len(bass)):
        if bars[i] != current_bar:
            count_bars += 1
            current_bar = bars[i]
        if beats[i] != current_beat:
            count_beats += current_beat - beats[i]
            current_beat = beats[i]
        
        
        
    
    

    
    
