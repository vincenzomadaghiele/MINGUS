#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:03:38 2021

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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

def structuredSongsToPM(structured_song, 
                        datasetToMidiChords, 
                        isJazz = False,
                        onlyChords = True):
    
    # input : a structured song json
    #structured_song = structuredSongs[0]
    #structured_song = new_structured_song
    beat_duration_sec = structured_song['beat duration [sec]']
    tempo = structured_song['tempo']
    
    
    # sampling of the measure
    unit = beat_duration_sec * 4 / 192.
    # possible note durations in seconds 
    # (it is possible to add representations - include 32nds, quintuplets...):
    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
    possible_durations = [unit * 192, unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3, 
                          unit * 144, unit * 72, unit * 36, unit * 18, 
                          unit * 32, unit * 16]

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
    dur_dict[possible_durations[11]] = 'half note triplet'
    dur_dict[possible_durations[12]] = 'quarter note triplet'
    inv_dur_dict = {v: k for k, v in dur_dict.items()}


    # Construct a PrettyMIDI object.
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Add a piano instrument
    inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    pm.instruments.append(inst)
    velocity = 95
    chord_velocity = 60
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
            
            if onlyChords:
                if beat['chord'] != last_chord:
                    # append last chord to pm inst
                    if last_chord != 'NC' and last_chord in datasetToMidiChords.keys():
                        for chord_pitch in datasetToMidiChords[last_chord][:3]:
                            inst.notes.append(pretty_midi.Note(chord_velocity, int(chord_pitch), chord_start, next_beat_sec - beat_duration_sec))
                    # update next chord start
                    last_chord = beat['chord']
                    chord_start = next_beat_sec - beat_duration_sec
            else:
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
                            inst.notes.append(pretty_midi.Note(chord_velocity, int(pitch[i]), start, end))
                        offset_sec += duration_sec
            
            beat_counter += 1
            next_beat_sec = (beat_counter + 1) * beat_duration_sec
    
    if onlyChords:
        # append last chord
        if last_chord != 'NC' and last_chord in datasetToMidiChords.keys():
            for chord_pitch in datasetToMidiChords[last_chord][:3]:
                inst.notes.append(pretty_midi.Note(velocity, int(chord_pitch), chord_start, next_beat_sec - beat_duration_sec))
       
    return pm


if __name__ == '__main__':

    # LOAD DATA
    if con.DATASET == 'WjazzDB':
        
        WjazzDB = dataset.WjazzDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                     con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
        
        #train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
        #val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getValidationData()
        #test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTestData()
        
        songs = WjazzDB.getOriginalSongDict()
        structuredSongs = WjazzDB.getStructuredSongs()
        vocabPitch, vocabDuration, vocabBeat, vocabOffset = WjazzDB.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = WjazzDB.getInverseVocabs()
        WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = WjazzDB.getChordDicts()

        for song in structuredSongs:
            try:
                isJazz = True
                title = song['title']
                performer = song['performer']
                print('Converting: ', performer, title)
                pm = structuredSongsToPM(song, WjazzToMidiChords, isJazz, onlyChords=True)
                pm.write('data/02_SeqAttn_data/chords/'+ title + '_' + performer + '.mid')
                pm = structuredSongsToPM(song, WjazzToMidiChords, isJazz, onlyChords=False)
                pm.write('data/02_SeqAttn_data/melody/'+ title + '_' + performer + '.mid')
            except:
                print('Not able to convert', performer, title)

    elif con.DATASET == 'NottinghamDB':
        
        NottinghamDB = dataset.NottinghamDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                     con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
        
        #train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
        #val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getValidationData()
        #test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTestData()
        
        songs = NottinghamDB.getOriginalSongDict()
        structuredSongs = NottinghamDB.getStructuredSongs()
        vocabPitch, vocabDuration, vocabBeat = NottinghamDB.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix = NottinghamDB.getInverseVocabs()
        NottinghamChords, NottinghamToMusic21, NottinghamToChordComposition, NottinghamToMidiChords = NottinghamDB.getChordDicts()

    elif con.DATASET == 'CustomDB':
        
        CustomDB = dataset.CustomDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                                    con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
            
        #train_pitch_batched, train_duration_batched, train_chord_batched, train_next_chord_batched, train_bass_batched, train_beat_batched, train_offset_batched  = CustomDB.getTrainingData()
        #val_pitch_batched, val_duration_batched, val_chord_batched, val_next_chord_batched, val_bass_batched, val_beat_batched, val_offset_batched  = CustomDB.getValidationData()
        #test_pitch_batched, test_duration_batched, test_chord_batched, test_next_chord_batched, test_bass_batched, test_beat_batched, test_offset_batched  = CustomDB.getTestData()

        songs = CustomDB.getOriginalSongDict()
        structuredSongs = CustomDB.getStructuredSongs()
        vocabPitch, vocabDuration, vocabBeat, vocabOffset = CustomDB.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = CustomDB.getInverseVocabs()
        WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = CustomDB.getChordDicts()

        for song in structuredSongs:
            isJazz = True
            title = song['title']
            #performer = song['performer']
            print('Converting: ', title)
            pm = structuredSongsToPM(song, WjazzToMidiChords, isJazz, onlyChords=True)
            pm.write('data/02_SeqAttn_data/NottinghamDB/chords/'+ title + '.mid')
            pm = structuredSongsToPM(song, WjazzToMidiChords, isJazz, onlyChords=False)
            pm.write('data/02_SeqAttn_data/NottinghamDB/melody/'+ title + '.mid')


