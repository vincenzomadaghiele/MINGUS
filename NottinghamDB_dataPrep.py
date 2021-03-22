#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:39:06 2021

@author: vincenzomadaghiele
for info about the note_seq : https://github.com/magenta/note-seq/blob/master/note_seq/protobuf/music.proto#L252
"""

import json
import numpy as np
#import music21 as m21
import glob
from note_seq import abc_parser


if __name__=="__main__":
    
    
    path = 'data/nottingham-dataset-master/ABC_cleaned/'
    abc = '*.abc'
    abcFiles = glob.glob(path + abc)
    songs = []
    for file in abcFiles:
        
        #file = 'data/nottingham-dataset-master/ABC_cleaned/ashover.abc'
        abcSongbook = abc_parser.parse_abc_tunebook_file(file)
        
        for abcSong in abcSongbook[0].values():
            # ensure there is only one time signiture
            # ensure that the song is in 4/4
            #abcSong = abcSongbook[0][8]
            time_signatures = abcSong.time_signatures
            tempos = abcSong.tempos
            if len(time_signatures) == 1 and len(tempos) <= 1 and time_signatures[0].numerator == 4 and time_signatures[0].denominator == 4:
        
                song = {}
                metadata = abcSong.sequence_metadata
                song['title'] = metadata.title
                song['total time [sec]'] = abcSong.total_time
                song['quantization [sec]'] = abcSong.quantization_info.steps_per_second
                song['quantization [beat]'] = abcSong.quantization_info.steps_per_quarter
                
                if not abcSong.tempos:
                    song['avgtempo'] = 120
                else:
                    song['avgtempo'] = abcSong.tempos[0]
        
                beat_duration_sec = 1 / (song['avgtempo'] / 60)
        
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
        
                chords_times = []
                for textannotation in abcSong.text_annotations:
                    if textannotation.annotation_type == 1:
                        chord = textannotation.text
                        chord_time = textannotation.time
                        chords_times.append([chord, chord_time])
        
        
                pitch_array = []
                duration_array = []
                offset_array = []
                chord_array = []
                bass_pitch_array = []
                beat_array = []
                bar_array = []
                
                for i in range(len(abcSong.notes)-1):
                    
                    note = abcSong.notes[i]
                    pitch_array.append(note.pitch)
                    if note.start_time:
                        duration_sec = note.end_time - note.start_time
                    else:
                        # first note does not have start time
                        duration_sec = note.end_time
                    # calculate distance from each duration
                    distance = np.abs(np.array(possible_durations) - duration_sec)
                    idx = distance.argmin()
                    duration_array.append(dur_dict[possible_durations[idx]])
                    beat_array.append(0)
                    
                    nochord = True
                    for j in range(len(chords_times)-1):
                        if chords_times[j+1][1] > note.start_time and chords_times[j][1] <= note.start_time:
                            chord_array.append(chords_times[j][0])
                            
                            nochord = False
                    
                    # check for rests
                    intra_note_time = abcSong.notes[i+1].start_time - abcSong.notes[i].end_time
                    # if the interval between notes is greater than the smallest duration ('16th')
                    # and smaller than the greatest duration ('full') then there is a rest
                    if intra_note_time >= possible_durations[5]:
                        # there is a rest!
                        
                        # handle the possibility of pauses longer than a full note
                        while intra_note_time > possible_durations[0]:
                            pitch_array.append('R')
                            # calculate distance from each duration
                            distance = np.abs(np.array(possible_durations) - intra_note_time)
                            idx = distance.argmin()
                            duration_array.append(dur_dict[possible_durations[idx]])
                            beat_array.append(0)
                            intra_note_time -= possible_durations[idx]
                            
                            for j in range(len(chords_times)-1):
                                if chords_times[j+1][1] > abcSong.notes[i].end_time and chords_times[j][1] <= abcSong.notes[i].end_time:
                                    chord_array.append(chords_times[j][0])
                                    
                                    nochord = False
                        
                        pitch_array.append('R')
                        # calculate distance from each duration
                        distance = np.abs(np.array(possible_durations) - intra_note_time)
                        idx = distance.argmin()
                        duration_array.append(dur_dict[possible_durations[idx]])
                        beat_array.append(0)
                        for j in range(len(chords_times)-1):
                            if chords_times[j+1][1] > abcSong.notes[i].end_time and chords_times[j][1] <= abcSong.notes[i].end_time:
                                chord_array.append(chords_times[j][0])
                                
                                nochord = False
                    
                    if nochord:
                        print('No chord at song %s, note %d' % (song['title'], i))
                
                # all these vector should have the same length
                # each element corresponds to a note event
                song['pitch'] = pitch_array
                song['duration'] = duration_array
                song['offset'] = offset_array
                song['chords'] = chord_array
                song['bass pitch'] = bass_pitch_array
                song['beats'] = beat_array # atm filled with zeros to not give problem in training
                song['beat duration [sec]'] = beat_duration_sec
                song['bars'] = bar_array
                
                songs.append(song)
    
    # split into train, validation and test
    songs_split = {}
    # train: 70% 
    songs_split['train'] = songs[:int(len(songs)*0.7)]
    # train: 10% 
    songs_split['validation'] = songs[int(len(songs)*0.7)+1:int(len(songs)*0.7)+1+int(len(songs)*0.1)]
    # train: 20%
    songs_split['test'] = songs[int(len(songs)*0.7)+1+int(len(songs)*0.1):]
    
    
    # Convert dict to JSON and SAVE IT
    with open('data/NottinghamDB.json', 'w') as fp:
        json.dump(songs_split, fp, indent=4)
                
                
                