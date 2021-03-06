#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:56:44 2021

@author: vincenzomadaghiele
"""

import sqlite3
import json
import numpy as np


if __name__=="__main__":
    
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect("data/wjazzd.db")
    solos_cur = con.cursor()
    
    songs = []
    structured_songs = []
    
    solos_cur.execute('SELECT * FROM solo_info') 
    for row in solos_cur:
        
        # select only songs in 4/4
        if row[14] == '4/4':
            song = {}
        
            song['performer'] = row[4]
            song['title'] = row[5]
            song['solo part'] = row[7] # number of solo in the original track recording
            song['instrument'] = row[8]
            song['style'] = row[9]
            song['avgtempo'] = row[10]
            song['chord changes'] = row[15]
            
            structured_song = {}
            structured_song['title'] = row[5]
            structured_song['tempo'] = row[10]
            structured_song['chord changes'] = row[15]
            structured_song['performer'] = row[4]
            structured_song['solo part'] = row[7] # number of solo in the original track recording
            structured_song['instrument'] = row[8]
            structured_song['style'] = row[9]

            bars = []
            beats = []
            
            pitch_array = []
            duration_array = []
            offset_array = [] # starting point of note/rest with respect to the bar [0,96]
            chord_array = []
            bass_pitch_array = []
            beat_array = []
            bar_array = []
            
            # FIX melid OF THIS SONG
            melid = row[1]
            onset = 0
            last_onset = 0
            bar_onset = 0
            beat_dur_array = []
            # SELECT ALL BEATS OF THIS melid
            beats_cur = con.cursor()
            beats_cur.execute("SELECT * FROM beats WHERE melid = %d" % melid)
            for beat_row in beats_cur:
                # FIX bar NUMBER AND chord, bass pitch
                bass_pitch = beat_row[8]
                bar = beat_row[3]
                beat = beat_row[4] 
                if beat_row[6] != '':
                    chord = beat_row[6]
                #beat_onset = beat_row[2]
                
                beat_pitch = []
                beat_duration = []
                beat_offset = []
                
                row_count = 0
                
                if bar != -1:
                    # SELECT ALL EVENTS IN THIS melid WITH THIS bar NUMBER AND beat NUMBER
                    events_cur = con.cursor()
                    events_cur.execute("SELECT * FROM melody WHERE melid = %d AND bar = %d AND beat = %d ORDER BY eventid" % (melid, bar, beat)) 
                    for event_row in events_cur:
                        
                        # duration count could be adjusted to take into account
                        # of swing timing and far smaller durations!!
                        beat_duration_sec = event_row[14]
                        beat_dur_array.append(beat_duration_sec)
                        
                        '''
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
                        '''
                        
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
                        
                        
                        # Detect rest by onset subtraction 
                        # and add to arrays before notes
                        
                        # check for rests
                        onset = event_row[2]
                        intra_note_time = onset - last_onset
                        # if the interval between notes is greater than the smallest duration ('16th note triplet')
                        # and smaller than the greatest duration ('full') then there is a rest
                        if intra_note_time >= possible_durations[5]:
                            # there is a rest!
                            
                            # handle the possibility of rests longer than a full note
                            while intra_note_time > possible_durations[0]:
                                # calculate distance from each duration
                                distance = np.abs(np.array(possible_durations) - intra_note_time)
                                idx = distance.argmin()
                                duration = dur_dict[possible_durations[idx]]
                                intra_note_time -= possible_durations[idx]
                                
                                offset = min(int(bar_onset / (beat_duration_sec * 4) * 96),96)
                                bar_onset += intra_note_time
                                
                                pitch_array.append('R')
                                duration_array.append(duration)
                                chord_array.append(chord)
                                bass_pitch_array.append(bass_pitch)
                                beat_array.append(beat)
                                bar_array.append(bar)
                                #velocity_array.append(velocity)                    
                                offset_array.append(offset)
                                
                                # append to structured song
                                beat_pitch.append('R')
                                beat_duration.append(duration)
                                #beat_offset.append(offset)
                                last_onset = onset + intra_note_time
                            
                            # calculate distance from each duration
                            distance = np.abs(np.array(possible_durations) - intra_note_time)
                            idx = distance.argmin()
                            duration = dur_dict[possible_durations[idx]]
                            
                            offset = min(int(bar_onset / (beat_duration_sec * 4) * 96),96)
                            bar_onset += intra_note_time

                            
                            pitch_array.append('R')
                            duration_array.append(duration)
                            chord_array.append(chord)
                            bass_pitch_array.append(bass_pitch)
                            beat_array.append(beat)
                            bar_array.append(bar)
                            #velocity_array.append(velocity)                    
                            offset_array.append(offset)
                            
                            # append to structured song
                            beat_pitch.append('R')
                            beat_duration.append(duration)
                            beat_offset.append(offset)
                            last_onset = onset + intra_note_time
                        
                        pitch = event_row[3]
                        duration_sec = event_row[4]
                        
                        distance = np.abs(np.array(possible_durations) - duration_sec)
                        idx = distance.argmin()
                        duration = dur_dict[possible_durations[idx]]
                        
                        offset = min(int(bar_onset / (beat_duration_sec * 4) * 96),96)
                        bar_onset += duration_sec
                        #velocity = 
                        
                        
                        pitch_array.append(pitch)
                        duration_array.append(duration)
                        chord_array.append(chord)
                        bass_pitch_array.append(bass_pitch)
                        beat_array.append(beat)
                        bar_array.append(bar)
                        #velocity_array.append(velocity)                    
                        offset_array.append(offset)
                        last_onset = onset + duration_sec
                        #print(last_onset)
                        
                        # append to structured song
                        beat_pitch.append(pitch)
                        beat_duration.append(duration)
                        beat_offset.append(offset)
                        row_count += 1

                    # if the list of events for this beat is empty
                    if row_count == 0:
                        # if there is no note going on:
                        # append rest to the beat
                        # update global onset
                        onset = beat_row[2]
                        if last_onset > onset + beat_duration_sec:
                            pass
                            # case 1: last note lasts more than a beat 
                            #duration_sec = last_onset - onset
                            #distance = np.abs(np.array(possible_durations) - duration_sec)
                            #idx = distance.argmin()
                            
                            #offset = int(bar_onset / (beat_duration_sec * 4) * 96)
                            #bar_onset += duration_sec
                            
                            #rest_duration = dur_dict[possible_durations[idx]]
                            #beat_pitch.append('R')
                            #beat_duration.append(rest_duration)
                            #beat_offset.append(offset)
                            #last_onset += duration_sec
                        elif last_onset > onset and last_onset < onset + beat_duration_sec:
                            # case 2: last note lasts less then one beat
                            # calculate last part of duration 
                            duration_sec = onset + beat_duration_sec - last_onset 
                            distance = np.abs(np.array(possible_durations) - duration_sec)
                            idx = distance.argmin()
                            
                            offset = min(int(bar_onset / (beat_duration_sec * 4) * 96),96)
                            bar_onset += duration_sec
                        
                            rest_duration = dur_dict[possible_durations[idx]]
                            
                            pitch_array.append('R')
                            duration_array.append(rest_duration)
                            chord_array.append(chord)
                            bass_pitch_array.append(bass_pitch)
                            beat_array.append(beat)
                            bar_array.append(bar)
                            #velocity_array.append(velocity)                    
                            offset_array.append(offset)

                            beat_pitch.append('R')
                            beat_duration.append(rest_duration)
                            beat_offset.append(offset)
                            
                            last_onset += duration_sec
                            
                        else:
                            # case 3: last note is finished
                            beat_pitch.append('R')
                            beat_duration.append('quarter')
                            
                            duration_sec = inv_dur_dict['quarter']
                            
                            offset = min(int(bar_onset / (beat_duration_sec * 4) * 96),96)
                            bar_onset += duration_sec
                            
                            pitch_array.append('R')
                            duration_array.append(rest_duration)
                            chord_array.append(chord)
                            bass_pitch_array.append(bass_pitch)
                            beat_array.append(beat)
                            bar_array.append(bar)
                            #velocity_array.append(velocity)                    
                            offset_array.append(offset)                            
                            
                            beat_offset.append(offset)
                            last_onset = onset + duration_sec
                            
                        # find remaining time from last played note
                        # put rest in the beat
                        # if there is no note remaining add quarter rest
                    
                    
                    
                    new_beat = {}
                    new_beat['num beat'] = beat 
                    new_beat['chord'] = chord 
                    new_beat['pitch'] = beat_pitch 
                    new_beat['duration'] = beat_duration 
                    new_beat['offset'] = beat_offset
                    new_beat['scale'] = []
                    new_beat['bass'] = bass_pitch
                    new_beat['this beat duration [sec]'] = beat_duration_sec
                    # append beat
                    beats.append(new_beat)
                    if beat == 4:
                        # append bar
                        new_bar = {}
                        new_bar['num bar'] = bar # over all song
                        new_bar['beats'] = beats # beats 1,2,3,4
                        bars.append(new_bar)
                        beats = []
                        bar_onset = 0

                        
            
            
            # it might be necessary to locate the position of the first bar in the standard
            # by extracting the chord sequence and comparing it (but atm is not useful)
            # also not useful for generation because later we go with the standard
            # but using a new function
            # ----
            # to solve: the rests are not in the right beat
            # if in a beat there are no events the rests are not counted!!!
            # when solving take into account last note duration:
            # not blindly put a rest on empty beats
            
            if bars:
                # Remove long rests from first beat
                new_pitch = []
                new_duration = []
                beat_offset = 0
                for i in range(len(bars[0]['beats'][0]['pitch'])):
                    if bars[0]['beats'][0]['pitch'][i] == 'R':
                        pitch_array.pop(0)
                        duration_array.pop(0)
                        offset_array.pop(0)
                    else:
                        new_pitch.append(bars[0]['beats'][0]['pitch'][i])
                        new_duration.append(bars[0]['beats'][0]['duration'][i])
                        beat_offset += inv_dur_dict[bars[0]['beats'][0]['duration'][i]]
                
                rest_duration_sec = bars[0]['beats'][0]['this beat duration [sec]'] - beat_offset
                distance = np.abs(np.array(possible_durations) - rest_duration_sec)
                idx = distance.argmin()
                new_rest_duration = dur_dict[possible_durations[idx]]
                new_pitch.insert (0, 'R')
                new_duration.insert (0, new_rest_duration) 
                
                # fix offset from first beat
                offset_pos = 0
                bar_onset = 0
                new_offset = []
                # first beat
                for i in range(len(new_pitch)):
                    duration_sec = inv_dur_dict[new_duration[i]]
                    offset = min(int(bar_onset / (bars[0]['beats'][0]['this beat duration [sec]'] * 4) * 96),96)
                    bar_onset += duration_sec
                    new_offset.append(offset)
                    offset_array[offset_pos] = offset
                    offset_pos += 1
                
                bars[0]['beats'][0]['pitch'] = new_pitch
                bars[0]['beats'][0]['duration'] = new_duration
                bars[0]['beats'][0]['offset'] = new_offset
    
                if len(bars[0]['beats']) > 1:
                    # second beat 
                    new_offset = []
                    # first beat
                    for i in range(len(bars[0]['beats'][1]['pitch'])):
                        duration_sec = inv_dur_dict[bars[0]['beats'][1]['duration'][i]]
                        offset = min(int(bar_onset / (bars[0]['beats'][1]['this beat duration [sec]'] * 4) * 96),96)
                        bar_onset += duration_sec
                        new_offset.append(offset)
                        offset_array[offset_pos] = offset
                        offset_pos += 1
                    bars[0]['beats'][1]['offset'] = new_offset
                
                if len(bars[0]['beats']) > 2:
                    # third beat 
                    new_offset = []
                    # first beat
                    for i in range(len(bars[0]['beats'][2]['pitch'])):
                        duration_sec = inv_dur_dict[bars[0]['beats'][2]['duration'][i]]
                        offset = min(int(bar_onset / (bars[0]['beats'][2]['this beat duration [sec]'] * 4) * 96),96)
                        bar_onset += duration_sec
                        new_offset.append(offset)
                        offset_array[offset_pos] = offset
                        offset_pos += 1
                    bars[0]['beats'][2]['offset'] = new_offset
                
                if len(bars[0]['beats']) > 3:
                    # fourth beat 
                    new_offset = []
                    # first beat
                    for i in range(len(bars[0]['beats'][3]['pitch'])):
                        duration_sec = inv_dur_dict[bars[0]['beats'][3]['duration'][i]]
                        offset = min(int(bar_onset / (bars[0]['beats'][3]['this beat duration [sec]'] * 4) * 96),96)
                        bar_onset += duration_sec
                        new_offset.append(offset)
                        offset_array[offset_pos] = offset
                        offset_pos += 1
                    bars[0]['beats'][3]['offset'] = new_offset
                
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
        
                
                # compute beats to next chord
                
                
                structured_song['bars'] = bars
                structured_song['beat duration [sec]'] = np.mean(beat_dur_array) 
    
                # all these vector should have the same length
                # each element corresponds to a note event
                song['pitch'] = pitch_array
                song['duration'] = duration_array
                song['offset'] = offset_array
                song['chords'] = chord_array
                song['next chords'] = next_chord_array
                song['bass pitch'] = bass_pitch_array
                song['beats'] = beat_array
                song['beat duration [sec]'] = np.mean(beat_dur_array)
                song['bars'] = bar_array
    
                # how to represent rest?
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
    with open('data/WjazzDB.json', 'w') as fp:
        json.dump(songs_split, fp, indent=4)
    
    
