#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:51:14 2021

@author: vincenzomadaghiele
"""

import pretty_midi
import music21 as m21
import json
import numpy as np
import torch
import torch.nn as nn
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

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

    # Convert to musicXML
    previousID = ''
    for structuredSong in structuredSongs:
    
        songID = structuredSong['performer'] + '_' + structuredSong['title']
        if songID == previousID:
            songID += '_2'
        
        # use 12 duration values
        duration_to_m21 = {} 
        duration_to_m21['full'] = 4
        duration_to_m21['half'] = 2
        duration_to_m21['quarter'] = 1
        duration_to_m21['8th'] = 1/2
        duration_to_m21['16th'] = 1/4
        duration_to_m21['32th'] = 1/8
        duration_to_m21['dot half'] = 3
        duration_to_m21['dot quarter'] = 1 + 1/2
        duration_to_m21['dot 8th'] = 1/2 + 1/4
        duration_to_m21['dot 16th'] = 1/4 + 1/8
        duration_to_m21['half note triplet'] = 4/3
    
        # create a new m21 stream
        m = m21.stream.Measure()
        tempo = m21.tempo.MetronomeMark(number=structuredSong['tempo'])
        m.timeSignature = m21.meter.TimeSignature('4/4')
        m.append(tempo)
        stream = m21.stream.Stream()
        previous_chord = 'NC'
        offset = 0
        for bar in structuredSong['bars']:
            for beat in bar['beats']:
                # append chord to the bar
                if beat['chord'] != previous_chord and beat['chord'] != 'NC':
                    m21chord = WjazzToMusic21[beat['chord']]
                    h = m21.harmony.ChordSymbol(m21chord)
                    m.insert(beat['num beat']-1, h)
                    previous_chord = beat['chord']
                # append notes to the bar
                for i in range(len(beat['pitch'])):
                    if beat['pitch'][i] == 'R':
                        new_note = m21.note.Rest(quarterLength=duration_to_m21[beat['duration'][i]])
                    else:  
                        new_note = m21.note.Note(midi=beat['pitch'][i], quarterLength=duration_to_m21[beat['duration'][i]])
                    m.append(new_note)
            m.number = bar['num bar']
            # append bar to the stream
            stream.append(m)
            m = m21.stream.Measure()
        # convert to xml
        try:
            xml_converter = m21.converter.subConverters.ConverterMusicXML()
            xml_converter.write(stream, 'musicxml', 'data/WjazzDBxml/' + songID + '.xml')
        except:
            pass
        previousID = songID
    


