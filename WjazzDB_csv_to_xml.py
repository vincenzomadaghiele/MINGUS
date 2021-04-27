#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:05:59 2021

@author: vincenzomadaghiele
"""
import music21 as m21
import pandas as pd
import numpy as np
import glob
import torch

import loadDBs as dataset
import MINGUS_const as con

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
    vocabPitch, vocabDuration, vocabBeat, vocabOffset = WjazzDB.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = WjazzDB.getInverseVocabs()
    WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = WjazzDB.getChordDicts()

    #%%
    possible_durations = [4, 2, 1, 1/2, 1/4, 1/8, 1/16, 
                          3, 3/2, 3/4, 3/8, 
                          1/6, 1/12]
    
    rests_durations = [4, 2, 1, 1/2, 1/4, 1/8,
                       3, 3/2, 3/4, 3/8,
                       1/6]


    #%% Load csv as dataframe
    
    source_path = 'data/WjazzDBcsv/csv_beats/*.csv'
    source_songs = glob.glob(source_path)
    source_songs = ['data/WjazzDBcsv/csv_beats/ArtPepper_Anthropology_Solo.csv']
    
    for csv_path in source_songs:
        
        beat_df = pd.read_csv(csv_path)
        song_name = csv_path[26:-4]
        melody_path = 'data/WjazzDBcsv/csv_melody/' + song_name + '.csv'
        meldoy_df = pd.read_csv(melody_path)
        print('converting ', song_name)
        #try:
        #beat_df = pd.read_csv("data/WjazzDBcsv/csv_beats/ArtPepper_Anthropology_Solo.csv")
        #meldoy_df = pd.read_csv("data/WjazzDBcsv/csv_melody/ArtPepper_Anthropology_Solo.csv")
        
        #tempo = 218
        
        tempo = int(60 / meldoy_df['beatdur'].mean())
        
        # get time signature
        time_signature = beat_df[beat_df['signature'].notnull()]['signature'].values[0]
        
        # remove empty beats at the beginning
        first_bar = max(beat_df.iloc[0]['bar'], meldoy_df.iloc[0]['bar'])
        meldoy_df = meldoy_df[(meldoy_df.bar >= first_bar)]
        beat_df = beat_df[(beat_df.bar >= first_bar)]
        
        # fill missing chords with NC
        beat_df['chord'] = beat_df['chord'].fillna('NC')
        
        # rescale onsets to start from first beat
        meldoy_df['onset'] = meldoy_df['onset'] - beat_df.iloc[0]['onset']
        beat_df['onset'] = beat_df['onset'] - beat_df.iloc[0]['onset']
        
        # create a new m21 stream
        stream = m21.stream.Stream()
        m = m21.stream.Measure()
        tempo = m21.tempo.MetronomeMark(number=tempo)
        m.append(tempo)
        m.timeSignature = m21.meter.TimeSignature('4/4')
        
        # constants for iteration
        current_bar = beat_df.iloc[0]['bar']
        current_bar_start_time = beat_df.iloc[0]['onset']
        counter96 = 0
        for i, row in beat_df.iterrows():
            # check for bar end
            if row['bar'] != current_bar:
                m.number = current_bar
                # append bar to the stream
                stream.append(m)
                m = m21.stream.Measure()
                current_bar = row['bar']
                current_bar_start_time = row['onset']
                #break
            # append chords
            if row['chord'] != 'NC':
                m21chord = WjazzToMusic21[row['chord']]
                h = m21.harmony.ChordSymbol(m21chord)
                m.insert(row['beat']-1, h)
            
            # find all notes in the bar
            if row['beat'] == 1:
                _thisBarNotes = meldoy_df[(meldoy_df.bar == current_bar)]
                thisBarNotes = _thisBarNotes.copy()[:2]
                # constants for iteration
                if counter96 == 0:
                    last_onset = current_bar_start_time
                for j, note in _thisBarNotes.iterrows():
                    # check for rests 
                    #offset96 = round((note['onset'] - current_bar_start_time) * 96)
                    #duration96 = round((note['duration'] / note['beatdur']) * 12)
                    rest_dur = (note['onset'] - last_onset) / note['beatdur']
                    if rest_dur > min(rests_durations):
                        
                        new_row = {
                            'bar': note['bar'], 
                            'beat': note['beat'], # check (useful?)
                            'beatdur': note['beatdur'], 
                            'duration': rest_dur,
                            'onset': last_onset,
                            'pitch': 'R',
                            }
                        
                        distance = np.abs(np.array(possible_durations) - rest_dur)
                        idx = distance.argmin()
                        duration = possible_durations[idx]
                        counter96 += duration * 24
                        #print(counter96)
                        # append to bar notes
                        thisBarNotes = thisBarNotes.append(new_row, ignore_index=True)
                        #last_onset += rest_dur
                        # append to stream measure
                        new_note = m21.note.Rest(quarterLength=duration)
                        m.append(new_note)
                    #else:
                        # aggregate duration to the note before!
                        #note['duration'] = note['duration'] + (note['onset'] - last_onset)
                    
                    dur = note['duration'] / note['beatdur']
                    distance = np.abs(np.array(possible_durations) - dur)
                    idx = distance.argmin()
                    duration = possible_durations[idx]
                    counter96 += duration * 24
                    
                    
                    # append to bar notes
                    thisBarNotes = thisBarNotes.append(note)
                    last_onset = note['onset'] + note['duration']
                    # append to stream measure
                    new_note = m21.note.Note(midi=note['pitch'], quarterLength=duration)
                    m.append(new_note)
                    
                if counter96 <= 96:
                    # aggregate to last note duration first and then rest
                    
                    # add rests
                    duration = min(((96 - counter96) * 4) / 96, 4)
                    #print('less')
                    new_note = m21.note.Rest(quarterLength=duration)
                    m.append(new_note)
                    counter96 = 0
                else:
                    #print('more')
                    counter96 -= 96
                    
                    #print(offset96, duration96, duration * 12)
            
                    # durations in a beat should sum to 96
                    # if a pitch exceeds no problem 
                    # calculate by how much and update next bar 96counter
                    
                thisBarNotes = thisBarNotes[2:].reset_index()
        
        xml_converter = m21.converter.subConverters.ConverterMusicXML()
        xml_converter.write(stream, 'musicxml', 'data/WjazzDBxml2/'+song_name+'.xml')
        #except:
            #print('Not able to convert ', song_name)
    


        
    