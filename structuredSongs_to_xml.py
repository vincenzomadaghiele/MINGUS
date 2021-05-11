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
import WjazzDB_csv_to_xml as wj

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

def NottinghamChordToM21(chord):
    new_chord_name = chord
    
    # special cases
    if chord == 'D/f+':
        new_chord_name = 'D/F#'
    if chord == 'G/b':
        new_chord_name = 'G/B' 
    if chord == 'C#d':
        new_chord_name = 'C#'
    if chord == 'G#d':
        new_chord_name = 'G#'
    if chord == 'D/a':
        new_chord_name = 'D/A' 
    if chord == 'A7/f+':
        new_chord_name = 'A7/F#'
    if chord == 'D7/b':
        new_chord_name = 'D7/B'
    if chord == 'D#d':
        new_chord_name = 'D#'
    if chord == 'Ad':
        new_chord_name = 'A'
    if chord == 'D7/f+':
        new_chord_name = 'D7/f#'
    if chord == 'Am/g':
        new_chord_name = 'Am/G'
    if chord == 'Gd':
        new_chord_name = 'G'
    if chord == 'E7/b':
        new_chord_name = 'E7/B'
    if chord == 'A/c+':
        new_chord_name = 'A/C#'
    if chord == 'Ed':
        new_chord_name = 'E'
    if chord == 'E7/g+':
        new_chord_name = 'E7/G#'
    if chord == 'G/d':
        new_chord_name = 'G/D'
    if chord == 'Bb':
        new_chord_name = 'B-'
    if chord == 'Eb':
        new_chord_name = 'E-' 
    if chord == 'Ab':
        new_chord_name = 'A-' 

    return new_chord_name

if __name__ == '__main__':

    '''
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
    '''
            
    # DATA LOADING
    print('Loading data from the Nottingham Database...')
    songs_path = 'data/NottinghamDB.json'
    songs_path = 'output/gen4eval_WjazzDB/original/WjazzDB_original.json'
    with open(songs_path) as f:
        songs = json.load(f)

    structuredSongs = songs #['structured for generation']

    # Convert to musicXML
    previousID = ''
    for structuredSong in structuredSongs:
    
        #songID = structuredSong['performer'] + '_' + structuredSong['title']
        songID = structuredSong['title']
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
        duration_to_m21['half note triplet'] = 1/6
        duration_to_m21['quarter note triplet'] = 1/12
    
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
                #if beat['chord'] != previous_chord and beat['chord'] != 'NC':
                    # append chord to the bar
                if beat['chord'] != previous_chord and beat['chord'] != 'NC':
                    #m21chord = WjazzToMusic21[beat['chord']]
                    #m21chord = NottinghamChordToM21(beat['chord'])
                    
                    m21chord = wj.WjazzChordToM21(beat['chord'])
                    try:
                        h = m21.harmony.ChordSymbol(m21chord)
                        m.insert(beat['num beat']-1, h)
                        previous_chord = beat['chord']
                    except:
                        print('not recognized chord', m21chord)
                # append notes to the bar
                for i in range(len(beat['pitch'])):
                    if beat['pitch'][i] == 'R':
                        new_note = m21.note.Rest(quarterLength=duration_to_m21[beat['duration'][i]])
                    else:  
                        new_note = m21.note.Note(midi=beat['pitch'][i], quarterLength=duration_to_m21[beat['duration'][i]])
                    m.append(new_note)
            m.number = bar['num bar']
            l = 0
            for note in m.notesAndRests:
                l += 1
            if l > 0:
                # append bar to the stream
                stream.append(m)
                m = m21.stream.Measure()
        # convert to xml
        #try:
        xml_converter = m21.converter.subConverters.ConverterMusicXML()
        xml_converter.write(stream, 'musicxml', 'output/gen4eval_WjazzDB/original/xml/' + songID + '.xml')
        #except:
            #pass
        previousID = songID
    


