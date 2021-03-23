#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:39:06 2021

@author: vincenzomadaghiele
for info about the note_seq : https://github.com/magenta/note-seq/blob/master/note_seq/protobuf/music.proto#L252

"""

import json
import numpy as np
import glob
from note_seq import abc_parser


if __name__=="__main__":
    
    path = 'data/nottingham-dataset-master/ABC_cleaned/'
    abc = '*.abc'
    abcFiles = glob.glob(path + abc)
    songs = []
    structured_songs = []
    for file in abcFiles:
        
        abcSongbook = abc_parser.parse_abc_tunebook_file(file)
        
        # iterate over all valid songs in the abc songbook
        # each song is a note_sequence element 
        for abcSong in abcSongbook[0].values():
            # ensure there is only one time signiture
            # ensure that the song is in 4/4

            time_signatures = abcSong.time_signatures
            tempos = abcSong.tempos
            if len(time_signatures) == 1 and len(tempos) <= 1 and time_signatures[0].numerator == 4 and time_signatures[0].denominator == 4:
        
                # DEFINE THE BASIC SONG PROPERTIES
                song = {}
                metadata = abcSong.sequence_metadata
                song['title'] = metadata.title
                song['total time [sec]'] = abcSong.total_time
                song['quantization [sec]'] = abcSong.quantization_info.steps_per_second
                song['quantization [beat]'] = abcSong.quantization_info.steps_per_quarter
                
                structured_song = {}
                structured_song['title'] = metadata.title
                structured_song['total time [sec]'] = abcSong.total_time
                structured_song['quantization [sec]'] = abcSong.quantization_info.steps_per_second
                structured_song['quantization [beat]'] = abcSong.quantization_info.steps_per_quarter
                
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
                inv_dur_dict = {v: k for k, v in dur_dict.items()}
        
                chords_times = []
                for textannotation in abcSong.text_annotations:
                    if textannotation.annotation_type == 1:
                        chord = textannotation.text
                        chord_time = textannotation.time
                        chords_times.append([chord, chord_time])
                
                
                # check if the song has chords
                if chords_times:
                    # BASIC ARRAYS FOR songs
                    pitch_array = []
                    duration_array = []
                    offset_array = []
                    chord_array = []
                    bass_pitch_array = []
                    beat_array = []
                    bar_array = []
                    
                    # BASIC ARRAYS FOR structured_songs
                    bars = []
                    bar_num = 0 # bar count starts from 0 
                    beats = []
                    # trick: only works for this particular dataset
                    if chords_times[0][1] != 0:
                        beat_num = 3
                    else:
                        beat_num = 0 # beat count is [0,3]
                    beat_pitch = []
                    beat_duration = []
                    offset_sec = 0
                    beat_counter = 0
                                    
                    next_beat_sec = (beat_counter + 1) * beat_duration_sec 
                    # iterate over the note_sequence notes
                    for i in range(len(abcSong.notes)):
                        
                        note = abcSong.notes[i]
                        pitch_array.append(note.pitch)
                        beat_pitch.append(note.pitch)
                        if note.start_time:
                            duration_sec = note.end_time - note.start_time
                        else:
                            # first note does not have start time
                            duration_sec = note.end_time
                        # calculate distance from each duration
                        distance = np.abs(np.array(possible_durations) - duration_sec)
                        idx = distance.argmin()
                        duration_array.append(dur_dict[possible_durations[idx]])
                        beat_duration.append(dur_dict[possible_durations[idx]])
                        offset_sec += duration_sec
                        beat_array.append(1)
                        
                        # check for chords
                        nochord = True
                        for j in range(len(chords_times)-1):
                            if chords_times[j+1][1] > note.start_time and chords_times[j][1] <= note.start_time:
                                chord = chords_times[j][0]
                                chord_array.append(chord)
                                nochord = False
                            elif chords_times[-1][1] <= note.start_time:
                                chord = chords_times[-1][0]
                                chord_array.append(chord)
                                nochord = False
                                break
                        if nochord:
                            print('No chord at song %s, note %d' % (song['title'], i))
                            chord = 'NC'
                            chord_array.append(chord)
                        
                        # calculate at which second there is a new beat
                        #next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec + first_note_start
                        while offset_sec >= next_beat_sec:
                            if beat_num >= 3:
                                # end of bar
                                # append beat
                                beat = {}
                                beat['num beat'] = beat_num + 1
                                # check for chords
                                nochord = True
                                for j in range(len(chords_times)-1):
                                    if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                        chord = chords_times[j][0]
                                        nochord = False
                                    elif chords_times[-1][1] < next_beat_sec:
                                        chord = chords_times[-1][0]
                                        nochord = False
                                        break
                                if nochord:
                                    chord = 'NC'
                                beat['chord'] = chord 
                                beat['pitch'] = beat_pitch 
                                beat['duration'] = beat_duration 
                                beat['offset'] = []
                                beat['scale'] = []
                                beat['bass'] = []
                                beats.append(beat)
                                beat_pitch = []
                                beat_duration = []
                                # append bar
                                bar = {}
                                bar['num bar'] = bar_num + 1 # over all song
                                bar['beats'] = beats # beats 1,2,3,4
                                bars.append(bar)
                                beats = []
                                beat_num = 0
                                bar_num += 1
                                beat_counter += 1
                                next_beat_sec = (beat_counter + 1) * beat_duration_sec 
                            else:
                                # end of beat
                                beat = {}
                                # number of beat in the bar [1,4]
                                beat['num beat'] = beat_num + 1
                                # at most one chord per beat
                                # check for chords
                                nochord = True
                                for j in range(len(chords_times)-1):
                                    if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                        chord = chords_times[j][0]
                                        nochord = False
                                    elif chords_times[-1][1] < next_beat_sec:
                                        chord = chords_times[-1][0]
                                        nochord = False
                                        break
                                if nochord:
                                    chord = 'NC'
                                beat['chord'] = chord 
                                # pitch of notes which START in this beat
                                beat['pitch'] = beat_pitch 
                                # duration of notes which START in this beat
                                beat['duration'] = beat_duration 
                                # offset of notes which START in this beat wrt the start of the bar
                                beat['offset'] = []
                                # get from chord with m21
                                beat['scale'] = []
                                beat['bass'] = []
                                # append beat
                                beats.append(beat)
                                beat_pitch = []
                                beat_duration = []
                                beat_num += 1
                                beat_counter += 1
                                next_beat_sec = (beat_counter + 1) * beat_duration_sec 
    
                        
                        # check for rests
                        
                        if i != len(abcSong.notes)-1:
                            intra_note_time = abcSong.notes[i+1].start_time - abcSong.notes[i].end_time
                            # if the interval between notes is greater than the smallest duration ('16th')
                            # and smaller than the greatest duration ('full') then there is a rest
                            if intra_note_time >= possible_durations[5]:
                                # there is a rest!
                                # handle the possibility of pauses longer than a full note
                                while intra_note_time > possible_durations[0]:
                                    
                                    pitch_array.append('R')
                                    beat_pitch.append('R')
                                    # calculate distance from each duration
                                    distance = np.abs(np.array(possible_durations) - intra_note_time)
                                    idx = distance.argmin()
                                    duration_array.append(dur_dict[possible_durations[idx]])
                                    beat_duration.append(dur_dict[possible_durations[idx]])
                                    offset_sec += duration_sec
                                    beat_array.append(1)
                                    intra_note_time -= possible_durations[idx]
                                    
                                    # check for chords
                                    nochord = True
                                    for j in range(len(chords_times)-1):
                                        if chords_times[j+1][1] > note.start_time and chords_times[j][1] <= note.start_time:
                                            chord = chords_times[j][0]
                                            chord_array.append(chord)
                                            nochord = False
                                        elif chords_times[-1][1] <= note.start_time:
                                            chord = chords_times[-1][0]
                                            chord_array.append(chord)
                                            nochord = False
                                            break
                                    if nochord:
                                        print('No chord at song %s, note %d' % (song['title'], i))
                                        chord = 'NC'
                                        chord_array.append(chord)
                                    
                                    # calculate at which second there is a new beat
                                    next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec 
                                    while offset_sec >= next_beat_sec:
                                        if beat_num == 3:
                                            # end of bar
                                            # append beat
                                            beat = {}
                                            beat['num beat'] = beat_num + 1
                                            # check for chords
                                            nochord = True
                                            for j in range(len(chords_times)-1):
                                                if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                                    chord = chords_times[j][0]
                                                    nochord = False
                                            if nochord:
                                                chord = 'NC'
                                            beat['chord'] = chord 
                                            beat['pitch'] = beat_pitch 
                                            beat['duration'] = beat_duration 
                                            beat['offset'] = []
                                            beat['scale'] = []
                                            beat['bass'] = []
                                            beats.append(beat)
                                            beat_pitch = []
                                            beat_duration = []
                                            # append bar
                                            bar = {}
                                            bar['num bar'] = bar_num # over all song
                                            bar['beats'] = beats # beats 1,2,3,4
                                            bars.append(bar)
                                            beats = []
                                            beat_num = 0
                                            bar_num += 1
                                            next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec 
        
                                        else: 
                                            # end of beat
                                            beat = {}
                                            # number of beat in the bar [1,4]
                                            beat['num beat'] = beat_num + 1
                                            # check for chords
                                            nochord = True
                                            for j in range(len(chords_times)-1):
                                                if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                                    chord = chords_times[j][0]
                                                    nochord = False
                                            if nochord:
                                                chord = 'NC'
                                            # at most one chord per beat
                                            beat['chord'] = chord 
                                            # pitch of notes which START in this beat
                                            beat['pitch'] = beat_pitch 
                                            # duration of notes which START in this beat
                                            beat['duration'] = beat_duration 
                                            # offset of notes which START in this beat wrt the start of the bar
                                            beat['offset'] = []
                                            # get from chord with m21
                                            beat['scale'] = []
                                            beat['bass'] = []
                                            # append beat
                                            beats.append(beat)
                                            beat_pitch = []
                                            beat_duration = []
                                            beat_num += 1
                                            next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec 
        
                                
                                
                                pitch_array.append('R')
                                beat_pitch.append('R')
                                # calculate distance from each duration
                                distance = np.abs(np.array(possible_durations) - intra_note_time)
                                idx = distance.argmin()
                                duration_array.append(dur_dict[possible_durations[idx]])
                                beat_duration.append(dur_dict[possible_durations[idx]])
                                offset_sec += duration_sec
                                beat_array.append(1)
                                # check for chords
                                nochord = True
                                for j in range(len(chords_times)-1):
                                    if chords_times[j+1][1] > note.start_time and chords_times[j][1] <= note.start_time:
                                        chord = chords_times[j][0]
                                        chord_array.append(chord)
                                        nochord = False
                                    elif chords_times[-1][1] <= note.start_time:
                                        chord = chords_times[-1][0]
                                        chord_array.append(chord)
                                        nochord = False
                                        break
                                if nochord:
                                    print('No chord at song %s, note %d' % (song['title'], i))
                                    chord = 'NC'
                                    chord_array.append(chord)
                                
                                # calculate at which second there is a new beat
                                next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec 
                                while offset_sec >= next_beat_sec:
                                    if beat_num == 3:
                                        # end of bar
                                        # append beat
                                        beat = {}
                                        beat['num beat'] = beat_num + 1
                                        # check for chords
                                        nochord = True
                                        for j in range(len(chords_times)-1):
                                            if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                                chord = chords_times[j][0]
                                                nochord = False
                                        if nochord:
                                            chord = 'NC'
                                        beat['chord'] = chord 
                                        beat['pitch'] = beat_pitch 
                                        beat['duration'] = beat_duration 
                                        beat['offset'] = []
                                        beat['scale'] = []
                                        beat['bass'] = []
                                        beats.append(beat)
                                        beat_pitch = []
                                        beat_duration = []
                                        # append bar
                                        bar = {}
                                        bar['num bar'] = bar_num # over all song
                                        bar['beats'] = beats # beats 1,2,3,4
                                        bars.append(bar)
                                        beats = []
                                        beat_num = 0
                                        bar_num += 1
                                        next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec 
        
                                    else: 
                                        # end of beat
                                        beat = {}
                                        # number of beat in the bar [1,4]
                                        beat['num beat'] = beat_num + 1
                                        # check for chords
                                        nochord = True
                                        for j in range(len(chords_times)-1):
                                            if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                                chord = chords_times[j][0]
                                                nochord = False
                                        if nochord:
                                            chord = 'NC'
                                        # at most one chord per beat
                                        beat['chord'] = chord 
                                        # pitch of notes which START in this beat
                                        beat['pitch'] = beat_pitch 
                                        # duration of notes which START in this beat
                                        beat['duration'] = beat_duration 
                                        # offset of notes which START in this beat wrt the start of the bar
                                        beat['offset'] = []
                                        # get from chord with m21
                                        beat['scale'] = []
                                        beat['bass'] = []
                                        # append beat
                                        beats.append(beat)
                                        beat_pitch = []
                                        beat_duration = []
                                        beat_num += 1
                                        next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec 
        
                    
                    # append last bar in case it has not 4 beats
                    if beats:
                        # end of bar
                        # append beat
                        beat = {}
                        beat['num beat'] = beat_num + 1
                        # check for chords
                        nochord = True
                        for j in range(len(chords_times)-1):
                            if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                chord = chords_times[j][0]
                                nochord = False
                            elif chords_times[-1][1] < next_beat_sec:
                                chord = chords_times[-1][0]
                                nochord = False
                                break
                        if nochord:
                            chord = 'NC'
                        beat['chord'] = chord 
                        beat['pitch'] = beat_pitch 
                        beat['duration'] = beat_duration 
                        beat['offset'] = []
                        beat['scale'] = []
                        beat['bass'] = []
                        beats.append(beat)
                        beat_pitch = []
                        beat_duration = []
                        # append bar
                        bar = {}
                        bar['num bar'] = bar_num + 1 # over all song
                        bar['beats'] = beats # beats 1,2,3,4
                        bars.append(bar)
                        beats = []
                        beat_num = 0
                        bar_num += 1
                        beat_counter += 1
                        next_beat_sec = (beat_counter + 1) * beat_duration_sec
                        
                    if len(chord_array) != len(pitch_array):
                        print('Error at song %s' % song['title'] )
                    
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
                    
                    structured_song['bars'] = bars
                    structured_song['beat duration [sec]'] = beat_duration_sec
                    
                    songs.append(song)
                    structured_songs.append(structured_song)
    
    # split into train, validation and test
    songs_split = {}
    # train: 70% 
    songs_split['train'] = songs[:int(len(songs)*0.7)]
    # train: 10% 
    songs_split['validation'] = songs[int(len(songs)*0.7)+1:int(len(songs)*0.7)+1+int(len(songs)*0.1)]
    # train: 20%
    songs_split['test'] = songs[int(len(songs)*0.7)+1+int(len(songs)*0.1):]
    # structured songs (ordered by bar and beats)
    songs_split['structured for generation'] = structured_songs
    
    
    # Convert dict to JSON and SAVE IT
    with open('data/NottinghamDB.json', 'w') as fp:
        json.dump(songs_split, fp, indent=4)
    
    
    
    
    '''
    # structured song data structure
    structured_songs = []
    structured_song = {}
    structured_song['title'] = metadata.title
    structured_song['total time [sec]'] = abcSong.total_time
    structured_song['quantization [sec]'] = abcSong.quantization_info.steps_per_second
    structured_song['quantization [beat]'] = abcSong.quantization_info.steps_per_quarter
    bars = []
    bar = {}
    bar['num bar'] # over all song
    bar['beats'] # beats 1,2,3,4
    beats = []
    beat = {}
    beat['num beat'] # number of beat in the bar [0,3]
    beat['chord'] # at most one chord per beat
    beat['pitch'] # pitch of notes which START in this beat
    beat['duration'] # duration of notes which START in this beat
    beat['offset'] # offset of notes which START in this beat wrt the start of the bar
    beat['scale'] # get from chord with m21
    beat['bass']
    beats.append(beat)
    bars.append(bar)
    structured_songs.append(structured_song)
    songs_split['structured for generation'] = structured_songs
    '''
                
                