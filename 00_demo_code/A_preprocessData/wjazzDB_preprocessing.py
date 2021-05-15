#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 18:37:11 2021

@author: vincenzomadaghiele
"""
import pandas as pd
import numpy as np
import glob
import json
import torch

import A_preprocessData.data_preprocessing as prep 

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
    
    possible_durations = [4, 2, 1, 1/2, 1/4, 1/8,
                          3, 3/2, 3/4,
                          1/6, 1/12]
    
    rests_durations = [4, 2, 1, 1/2, 1/4, 1/8,
                       3, 3/2, 3/4,
                       1/6]

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
    dur_dict[possible_durations[9]] = 'half note triplet'
    dur_dict[possible_durations[10]] = 'quarter note triplet'

    
    source_path = 'A_preprocessData/data/WjazzDBcsv/csv_beats/*.csv'
    source_songs = glob.glob(source_path)
    #source_songs = ["data/WjazzDBcsv/csv_beats/DonEllis_YouSteppedOutOfADream-2_Solo.csv"]
    #source_songs = ["A_preprocessData/data/WjazzDBcsv/csv_beats/DonEllis_YouSteppedOutOfADream-2_Solo.csv"]
    
    structuredSongs = []
    songs = []
    for csv_path in source_songs:
        
        beat_df = pd.read_csv(csv_path)
        song_name = '.'.join(csv_path.split('/')[-1].split('.')[:-1])
        melody_path = 'A_preprocessData/data/WjazzDBcsv/csv_melody/' + song_name + '.csv'
        #melody_path = 'data/WjazzDBcsv/csv_melody/' + song_name + '.csv'
        meldoy_df = pd.read_csv(melody_path)
        print('converting ', song_name)
        
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

            new_structured_song = {}
            new_structured_song['performer'] = csv_path.split('/')[-1].split('.')[0].split('_')[0]
            new_structured_song['title'] = csv_path.split('/')[-1].split('.')[0].split('_')[1]
            new_structured_song['tempo'] = tempo
            new_structured_song['beat duration [sec]'] = 60 / new_structured_song['tempo']

            bar_num = 0
            bars = []
            beats = []
            beat_pitch = []
            beat_duration = []
            beat_offset = []
            chord = 'NC'
            bass = 'R'
            
            # constants for iteration
            current_bar = beat_df.iloc[0]['bar']
            current_bar_start_time = beat_df.iloc[0]['onset']
            counter96 = 0
            bar_offset = 0
            chord_counter = 0 
            if not beat_df['chord'][(beat_df.chord != 'NC')].empty and not beat_df['bass_pitch'].empty:
                last_chord = beat_df['chord'][(beat_df.chord != 'NC')].iloc[0]
                for i, row in beat_df.iterrows():
                    
                    # find all notes in the bar
                    if row['beat'] == beat_df.iloc[0]['beat']:
                        
                        beats = []
                        new_beat = {}
                        new_beat['num beat'] = row['beat']
                        #new_beat['chord'] = chord
                        new_beat['scale'] = []
                        #new_beat['bass'] = bass
                        new_beat['this beat duration [sec]'] = new_structured_song['beat duration [sec]']
                        beat_pitch = []
                        beat_duration = []
                        beat_offset = []

                        
                        if i+4 <= beat_df.shape[0] - 1:
                            next_bar_onset = beat_df.iloc[i+4]['onset']
                        _thisBarNotes = meldoy_df[(meldoy_df.bar == row['bar'])].reset_index()
                        
                        bar_duration = 0
                        
                        # update last onset
                        if bar_offset == 0:
                            last_onset = current_bar_start_time
                        if not _thisBarNotes.empty:
                            # check for rests at the start of the bar
                            rest_dur = (_thisBarNotes.iloc[0]['onset'] - last_onset) / _thisBarNotes.iloc[0]['beatdur']
                            if rest_dur > min(rests_durations):
                                distance = np.abs(np.array(rests_durations) - rest_dur)
                                idx = distance.argmin()
                                quarterDuration = rests_durations[idx]
                                counter96 += quarterDuration * 24
                                
                                duration = dur_dict[quarterDuration]
                                offset = int(bar_duration * 96 / 4)
                                pitch = 'R'
                                
                                beat_pitch.append(pitch)
                                beat_duration.append(duration)
                                beat_offset.append(offset)
                                
                                bar_duration += quarterDuration
                                
                            for j, note in _thisBarNotes.iterrows():
                                # check if beat has ended
                                if note['beat'] != new_beat['num beat']:
                                    
                                    new_beat['pitch'] = beat_pitch 
                                    new_beat['duration'] = beat_duration 
                                    new_beat['offset'] = beat_offset
                                    beats.append(new_beat)

                                    bt_num = new_beat['num beat'] + 1
                                    while bt_num < note['beat']:
                                        new_beat = {}
                                        new_beat['num beat'] = bt_num
                                        new_beat['scale'] = []
                                        new_beat['this beat duration [sec]'] = new_structured_song['beat duration [sec]']
                                        new_beat['pitch'] = [] 
                                        new_beat['duration'] = [] 
                                        new_beat['offset'] = []
                                        beats.append(new_beat)
                                        bt_num += 1
                                    
                                    new_beat = {}
                                    new_beat['num beat'] = note['beat']
                                    new_beat['scale'] = []
                                    new_beat['this beat duration [sec]'] = new_structured_song['beat duration [sec]']
                                    beat_pitch = []
                                    beat_duration = []
                                    beat_offset = []
                                    
                                # check if this is the last note in the bar
                                nextRest = 'no rest'
                                if j < _thisBarNotes.shape[0] - 1:
                                    # check for onset difference between this note and the next one
                                    rest_dur = (_thisBarNotes.iloc[j+1]['onset'] - note['onset'] - note['duration']) / note['beatdur']
                                    if rest_dur > min(rests_durations):
                                        # if rest append after the note
                                        distance = np.abs(np.array(rests_durations) - rest_dur)
                                        idx = distance.argmin()
                                        quarterDuration = rests_durations[idx]
                                        counter96 += quarterDuration * 24
                                        
                                        duration = dur_dict[quarterDuration]
                                        pitch = note['pitch']
                                        
                                        nextRest = ['R', duration, quarterDuration]
                                    else:
                                        # if no rest sum duration to the note 
                                        note['duration'] = _thisBarNotes.iloc[j+1]['onset'] - note['onset']
                                    
                                    
                                    # compute note duration
                                    dur = note['duration'] / note['beatdur']
                                    distance = np.abs(np.array(possible_durations) - dur)
                                    idx = distance.argmin()
                                    quarterDuration = possible_durations[idx]
                                    counter96 += quarterDuration * 24
                                    
                                    duration = dur_dict[quarterDuration]
                                    offset = int(bar_duration * 96 / 4)
                                    pitch = note['pitch']
                                    
                                    beat_pitch.append(pitch)
                                    beat_duration.append(duration)
                                    beat_offset.append(offset)
                                    
                                    bar_duration += quarterDuration

                                    # append note THEN rest to array
                                    if nextRest != 'no rest':
                                        beat_pitch.append(nextRest[0])
                                        beat_duration.append(nextRest[1])
                                        offset = int(bar_duration * 96 / 4)
                                        beat_offset.append(offset)
                                        bar_duration += nextRest[2]
                                        
                                else:                                    
                                    rest_dur = (next_bar_onset - (note['onset'] + note['duration'])) / note['beatdur']
                                    #print(next_bar_onset - (note['onset'] + note['duration']))
                                    if rest_dur > min(rests_durations):
                                        # if rest append after the note
                                        distance = np.abs(np.array(rests_durations) - rest_dur)
                                        idx = distance.argmin()
                                        quarterDuration = rests_durations[idx]
                                        counter96 += quarterDuration * 24
                                        
                                        duration = dur_dict[quarterDuration]
                                        offset = int(bar_duration * 96 / 4)
                                        pitch = note['pitch']
                                        
                                        nextRest = ['R', duration, offset, quarterDuration]
                                    else:
                                        # if no rest sum duration to the note 
                                        note['duration'] = next_bar_onset - note['onset']
                                    
                                    
                                    # this is the last note in the bar
                                    dur = note['duration'] / note['beatdur']
                                    distance = np.abs(np.array(possible_durations) - dur)
                                    idx = distance.argmin()
                                    quarterDuration = possible_durations[idx]
                                    counter96 += quarterDuration * 24
                                    
                                    duration = dur_dict[quarterDuration]
                                    offset = int(bar_duration * 96 / 4)
                                    pitch = note['pitch']
                                    
                                    beat_pitch.append(pitch)
                                    beat_duration.append(duration)
                                    beat_offset.append(offset)
                                    
                                    bar_duration += quarterDuration
                                    
                                    last_onset = note['onset'] + note['duration']
                                    #counter96 = 0
                        else:
                            # reset to 0 if a bar is skipped
                            counter96 = 0
                        
                        if bar_duration < 4:
                            dur = 4 - bar_duration
                            distance = np.abs(np.array(possible_durations) - dur)
                            idx = distance.argmin()
                            quarterDuration = possible_durations[idx]
                            counter96 += quarterDuration * 24
                            
                            duration = dur_dict[quarterDuration]
                            offset = int(bar_duration * 96 / 4)
                            pitch = 'R'
                            
                            beat_pitch.append(pitch)
                            beat_duration.append(duration)
                            beat_offset.append(offset)
        
                            bar_offset = 0
                    
                        # append last beat
                        new_beat['pitch'] = beat_pitch 
                        new_beat['duration'] = beat_duration 
                        new_beat['offset'] = beat_offset
                        beats.append(new_beat)
                    
                    # add last beats 
                    bt = len(beats) + 1
                    while len(beats) < 4:
                        new_beat = {}
                        new_beat['num beat'] = bt
                        new_beat['scale'] = []
                        new_beat['this beat duration [sec]'] = new_structured_song['beat duration [sec]']
                        new_beat['pitch'] = [] 
                        new_beat['duration'] = [] 
                        new_beat['offset'] = []
                        beats.append(new_beat)
                        bt += 1
                        
                    if len(beats) > 4:
                        beats = beats[:4]
                    
                    # append chords
                    if row['chord'] != 'NC':
                        m21chord = WjazzChordToM21(row['chord'])
                        # Handle exceptions
                        chord_components = [char for char in m21chord]
                        m21chord = ''
                        for kk in range(len(chord_components)):
                            m21chord += chord_components[kk]
                            if kk != len(chord_components) - 1:
                                if chord_components[kk] == '7' and (chord_components[kk+1] != 'b' or chord_components[kk+1] != '#'):
                                    break
                                if chord_components[kk] == '6' and (chord_components[kk+1] != 'b' or chord_components[kk+1] != '#'):
                                    m21chord = m21chord[:-1]
                                    break
                        
                        if row['beat']-1 < 4 :
                            #beats[row['beat']-1]['chord'] = row['chord']
                            beats[row['beat']-1]['chord'] = m21chord
                            last_chord = row['chord']
                            last_chord = m21chord
                    else:
                        if row['beat']-1 < 4 :
                            beats[row['beat']-1]['chord'] = last_chord

                    if row['beat']-1 < 4 :
                        # append bass
                        if row['bass_pitch']:
                            beats[row['beat']-1]['bass'] = row['bass_pitch']
                        else:
                            beats[row['beat']-1]['bass'] = 'R'
                    
                    if row['beat'] == 4:
                        # append bar
                        new_bar = {}
                        new_bar['num bar'] = row['bar'] # over all song
                        new_bar['beats'] = beats # beats 1,2,3,4
                        bars.append(new_bar)
                        beats = []
                        bar_num += 1
                        current_bar_start_time = row['onset']
                        #if bar_num == 4:
                        #    break
            
                # compute chords array
                chord_array = []
                for bar in bars:
                    for beat in bar['beats']:
                        if 'chord' not in beat.keys():
                            beat['chord'] = 'NC'
                        if 'bass' not in beat.keys():
                            beat['bass'] = 'R'
                        chord_array.append(beat['chord'])
                
                # compute next chord 
                last_chord = chord_array[0]
                next_chords = []
                for i in range(len(chord_array)):
                    if chord_array[i] != last_chord:
                        next_chords.append(chord_array[i])
                        last_chord = chord_array[i]
                
                # compute array of next chords
                next_chords.append('NC')
                next_chord_array = []
                next_chord_pointer = 0
                last_chord = chord_array[0]
                for i in range(len(chord_array)):
                    if chord_array[i] != last_chord:
                        last_chord = chord_array[i]
                        next_chord_pointer += 1
                    next_chord_array.append(next_chords[next_chord_pointer])
                
                
                # compute next chord 
                last_chord = bars[0]['beats'][0]['chord']
                next_chords2 = []
                for bar in bars:
                    for beat in bar['beats']:
                        if beat['chord'] != last_chord:
                            next_chords2.append(beat['chord'])
                            last_chord = beat['chord']
                
                # add next chord to the beats
                last_chord = bars[0]['beats'][0]['chord']
                next_chords2.append('NC')
                next_chord_pointer = 0
                for bar in bars:
                    for beat in bar['beats']:
                        if beat['chord'] != last_chord:
                            last_chord = beat['chord']
                            next_chord_pointer += 1
                        beat['next chord'] = next_chords2[next_chord_pointer]
    
                new_structured_song['bars'] = bars
                tune = prep.arraysFromStructuredSong(new_structured_song)
                structuredSongs.append(new_structured_song)
                songs.append(tune)
            
    # split into train, validation and test
    songs_split = {}
    # train: 70% 
    songs_split['train'] = songs[:int(len(songs)*0.7)]
    # train: 10% 
    songs_split['validation'] = songs[int(len(songs)*0.7)+1:int(len(songs)*0.7)+1+int(len(songs)*0.1)]
    # train: 20%
    songs_split['test'] = songs[int(len(songs)*0.7)+1+int(len(songs)*0.1):]
    # structured songs (ordered by bar and beats)
    songs_split['structured for generation'] = structuredSongs
    
    # Convert dict to JSON and SAVE IT
    with open('A_preprocessData/data/DATA.json', 'w') as fp:
        json.dump(songs_split, fp, indent=4)




    