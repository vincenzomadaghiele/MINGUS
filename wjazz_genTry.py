#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:58:37 2021

@author: vincenzomadaghiele
"""
import pretty_midi
import json
import numpy as np
import torch
import torch.nn as nn
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con
import MINGUS_condGenerate as gen

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
    structuredSongs = WjazzDB.getStructuredSongs()
    vocabPitch, vocabDuration, vocabBeat = WjazzDB.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix = WjazzDB.getInverseVocabs()
    WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = WjazzDB.getChordDicts()

    
    #%% 

    '''
    structured_song = structuredSongs[0]
    # input : a structured song json
    #structured_song = structuredSongs[0]
    #structured_song = new_structured_song
    title = structured_song['title']
    beat_duration_sec = structured_song['beat duration [sec]']
    tempo = structured_song['tempo']
    
    # sampling of the measure
    unit = beat_duration_sec * 4 / 96.
    # possible note durations in seconds 
    # (it is possible to add representations - include 32nds, quintuplets...):
    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
    possible_durations = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3,
                          unit * 72, unit * 36, unit * 18, unit * 9, 
                          unit * 32, unit * 16, unit * 8, unit * 4]

    # Define durations dictionary
    dur_dict = {}
    dur_dict[possible_durations[0]] = 'full'
    dur_dict[possible_durations[1]] = 'half'
    dur_dict[possible_durations[2]] = 'quarter'
    dur_dict[possible_durations[3]] = '8th'
    dur_dict[possible_durations[4]] = '16th'
    dur_dict[possible_durations[5]] = '32th'
    dur_dict[possible_durations[6]] = 'dot half'
    dur_dict[possible_durations[7]] = 'dot quarter'
    dur_dict[possible_durations[8]] = 'dot 8th'
    dur_dict[possible_durations[9]] = 'dot 16th'
    dur_dict[possible_durations[10]] = 'half note triplet'
    dur_dict[possible_durations[11]] = 'quarter note triplet'
    dur_dict[possible_durations[12]] = '8th note triplet'
    dur_dict[possible_durations[13]] = '16th note triplet'
    inv_dur_dict = {v: k for k, v in dur_dict.items()}
    

    # Construct a PrettyMIDI object.
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Add a piano instrument
    inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    chords_inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    pm.instruments.append(inst)
    pm.instruments.append(chords_inst)
    velocity = 90    
    offset_sec = 0
    beat_counter = 0
    next_beat_sec = (beat_counter + 1) * beat_duration_sec
    last_chord = structured_song['bars'][0]['beats'][0]['chord']
    chord_start = 0
    
    for bar in structured_song['bars']:
        
        if bar['beats'][0]['num beat'] != 1:
            beat_counter += bar['beats'][0]['num beat'] - 1
            offset_sec += (bar['beats'][0]['num beat'] - 1) * beat_duration_sec
            next_beat_sec = (beat_counter + 1) * beat_duration_sec
        
        for beat in bar['beats']:
            
            if beat['chord'] != last_chord:
                # append last chord to pm inst
                if last_chord != 'NC':
                    for chord_pitch in WjazzToMidiChords[last_chord][:3]:
                        chords_inst.notes.append(pretty_midi.Note(velocity, int(chord_pitch), chord_start, next_beat_sec - beat_duration_sec))
                # update next chord start
                last_chord = beat['chord']
                chord_start = next_beat_sec - beat_duration_sec

            pitch = beat['pitch']
            duration = beat['duration']
            for i in range(len(pitch)):
                if pitch[i] != '<pad>' and duration[i] != '<pad>':
                    if pitch[i] == 'R':
                        duration_sec = inv_dur_dict[duration[i]]
                    else:
                        duration_sec = inv_dur_dict[duration[i]]
                        start = offset_sec
                        end = offset_sec + duration_sec
                        inst.notes.append(pretty_midi.Note(velocity, int(pitch[i]), start, end))
                    offset_sec += duration_sec
            
            beat_counter += 1
            next_beat_sec = (beat_counter + 1) * beat_duration_sec
    
    # append last chord
    if last_chord != 'NC':
        for chord_pitch in WjazzToMidiChords[last_chord][:3]:
            chords_inst.notes.append(pretty_midi.Note(velocity, int(chord_pitch), chord_start, next_beat_sec - beat_duration_sec))
    '''
    
    new_structured_song = structuredSongs[2]
    title = new_structured_song['title']
    pm = gen.structuredSongsToPM(new_structured_song, WjazzToMidiChords)
    pm.write('output/'+ title + '.mid')
    
    