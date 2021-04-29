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

def WjazzChordToM21(chord):

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
    
    return new_chord_name


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
                       3, 3/2, 3/4, 
                       1/6]


    #%% Load csv as dataframe
    
    source_path = 'data/WjazzDBcsv/csv_beats/*.csv'
    source_songs = glob.glob(source_path)
    #source_songs = ["data/WjazzDBcsv/csv_beats/JoshuaRedman_HomeFries_Solo.csv"]
    
    for csv_path in source_songs:
        
        beat_df = pd.read_csv(csv_path)
        song_name = csv_path[26:-4]
        melody_path = 'data/WjazzDBcsv/csv_melody/' + song_name + '.csv'
        meldoy_df = pd.read_csv(melody_path)
        print('converting ', song_name)
        #try:
        if 0 == 0:
            #beat_df = pd.read_csv("data/WjazzDBcsv/csv_beats/ArtPepper_Anthropology_Solo.csv")
            #meldoy_df = pd.read_csv("data/WjazzDBcsv/csv_melody/ArtPepper_Anthropology_Solo.csv")
            
            #tempo = 218
            
            time_signature = beat_df[beat_df['signature'].notnull()]['signature'].values[0]

            if time_signature == '4/4':
                
                tempo = int(60 / meldoy_df['beatdur'].mean())
                
                # get time signature
                
                # remove empty beats at the beginning
                first_bar = max(beat_df.iloc[0]['bar'], meldoy_df.iloc[0]['bar'])
                meldoy_df = meldoy_df[(meldoy_df.bar >= first_bar)].reset_index()
                beat_df = beat_df[(beat_df.bar >= first_bar)].reset_index()
                
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
                bar_offset = 0
                chord_counter = 0 
                notes = []
                if not beat_df['chord'][(beat_df.chord != 'NC')].empty:
                    last_chord = beat_df['chord'][(beat_df.chord != 'NC')].iloc[0]
                    for i, row in beat_df.iterrows():
                        
                        # check for bar end
                        if row['bar'] != current_bar:
                            if chord_counter == 0 and last_chord != 'NC':
                                if last_chord in WjazzToMusic21.keys():
                                    m21chord = WjazzToMusic21[last_chord]
                                else:
                                    m21chord = WjazzChordToM21(last_chord)
                                h = m21.harmony.ChordSymbol(m21chord)
                                m.insert(0, h)
                            m.number = current_bar
                            # append bar to the stream
                            stream.append(m)
                            m = m21.stream.Measure()
                            current_bar = row['bar']
                            current_bar_start_time = row['onset']
                            chord_counter = 0
                            #if row['bar'] == 34:
                                #break
                            #break
                        
                        
                        # find all notes in the bar
                        if row['beat'] == 1:
                            
                            if i+4 <= beat_df.shape[0] - 1:
                                next_bar_onset = beat_df.iloc[i+4]['onset']
                            
                            _thisBarNotes = meldoy_df[(meldoy_df.bar == current_bar)].reset_index()
                            #thisBarNotes = _thisBarNotes.copy()[:2]
                            notes = []
                            
                            # update last onset
                            if bar_offset == 0:
                                last_onset = current_bar_start_time
            
                            if not _thisBarNotes.empty:
                                # check for rests at the start of the bar
                                rest_dur = (_thisBarNotes.iloc[0]['onset'] - last_onset) / _thisBarNotes.iloc[0]['beatdur']
                                if rest_dur > min(rests_durations):
                                    distance = np.abs(np.array(rests_durations) - rest_dur)
                                    idx = distance.argmin()
                                    duration = rests_durations[idx]
                                    counter96 += duration * 24
                                    notes.append(['R', duration])
                                for j, note in _thisBarNotes.iterrows():
                                    nextRest = 'no rest'
                                    # check that this is the last note in the bar
                                    if j < _thisBarNotes.shape[0] - 1:
                                        # check for onset difference between this note and the next one
                                        rest_dur = (_thisBarNotes.iloc[j+1]['onset'] - note['onset'] - note['duration']) / note['beatdur']
                                        if rest_dur > min(rests_durations):
                                            # if rest append after the note
                                            distance = np.abs(np.array(rests_durations) - rest_dur)
                                            idx = distance.argmin()
                                            duration = rests_durations[idx]
                                            counter96 += duration * 24
                                            nextRest = ['R', duration]
                                            #print(_thisBarNotes.iloc[j+1]['onset'], ' - ', note['onset'])
                                            #print('R', _thisBarNotes.iloc[j+1]['onset'] - note['onset'] - note['duration'], duration)
                                        else:
                                            # if no rest sum duration to the note 
                                            note['duration'] = _thisBarNotes.iloc[j+1]['onset'] - note['onset']
                                        
                                        
                                        # compute note duration
                                        dur = note['duration'] / note['beatdur']
                                        distance = np.abs(np.array(possible_durations) - dur)
                                        idx = distance.argmin()
                                        duration = possible_durations[idx]
                                        #print(note['pitch'],note['duration'], duration)
                                        counter96 += duration * 24
                                        
                                        # append note THEN rest to array
                                        notes.append([note['pitch'], duration])
                                        if nextRest != 'no rest':
                                            notes.append(nextRest)
                                    else:
                                        
                                        
                                        rest_dur = (next_bar_onset - (note['onset'] + note['duration'])) / note['beatdur']
                                        #print(next_bar_onset - (note['onset'] + note['duration']))
                                        if rest_dur > min(rests_durations):
                                            # if rest append after the note
                                            distance = np.abs(np.array(rests_durations) - rest_dur)
                                            idx = distance.argmin()
                                            duration = rests_durations[idx]
                                            counter96 += duration * 24
                                            nextRest = ['R', duration]
                                        else:
                                            # if no rest sum duration to the note 
                                            note['duration'] = next_bar_onset - note['onset']
                                        
                                        
                                        # this is the last note in the bar
                                        dur = note['duration'] / note['beatdur']
                                        distance = np.abs(np.array(possible_durations) - dur)
                                        idx = distance.argmin()
                                        duration = possible_durations[idx]
                                        counter96 += duration * 24
                                        notes.append([note['pitch'], duration])
                                        last_onset = note['onset'] + note['duration']
                                        #counter96 = 0
                                '''
                                # check if the bar is complete
                                if counter96 <= 96:           
                                    print(counter96)
                                    # add rests
                                    duration = min(((96 - counter96) * 4) / 96, 4)
                                    notes.append(['R', duration])
                                    counter96 = 0
                                else:
                                    counter96 -= 96
                                '''
                            else:
                                # reset to 0 if a bar is skipped
                                counter96 = 0
                                
                            # write to stream measure
                            bar_offset = 0
                            for note in notes:
                                if note[0] == 'R':
                                    if bar_offset + note[1] <= 4:
                                        new_note = m21.note.Rest(quarterLength=note[1])
                                        m.insert(bar_offset, new_note)
                                        bar_offset += note[1]
                                else:
                                    if bar_offset + note[1] <= 4:
                                        new_note = m21.note.Note(midi=note[0], quarterLength=note[1])
                                        m.insert(bar_offset, new_note)
                                        bar_offset += note[1]
                            #print(bar_offset)
                            if bar_offset < 4:
                                new_note = m21.note.Rest(quarterLength = 4-bar_offset)
                                m.insert(bar_offset, new_note)
                                bar_offset = 0
                                
                        # append chords
                        if row['chord'] != 'NC':
                            if row['chord'] in WjazzToMusic21.keys():
                                m21chord = WjazzToMusic21[row['chord']]
                                #print(row['chord'], m21chord)
                            else:
                                m21chord = WjazzChordToM21(row['chord'])
                            # Handle exceptions
                            chord_components = [char for char in m21chord]
                            m21chord = ''
                            for kk in range(len(chord_components)):
                                m21chord += chord_components[kk]
                                if kk != len(chord_components) - 1:
                                    if chord_components[kk] == '7' and (chord_components[kk+1] != 'b' or chord_components[kk+1] != '#'):
                                        break
                            
                            h = m21.harmony.ChordSymbol(m21chord)
                            m.insert(row['beat']-1, h)
                            chord_counter += 1
                            last_chord = row['chord']
    
                            
                            '''
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
                            '''
                            
                    xml_converter = m21.converter.subConverters.ConverterMusicXML()
                    xml_converter.write(stream, 'musicxml', 'data/WjazzDBxml5/'+song_name+'.xml')
        #except:
        #    print('Not able to convert ', song_name)
    


        
    