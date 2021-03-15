#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:56:44 2021

@author: vincenzomadaghiele
"""
import sqlite3
import numpy as np

# Create a SQL connection to our SQLite database
con = sqlite3.connect("data/wjazzd.db")
solos_cur = con.cursor()

songs = []

#songs['train'] = []
#songs['validation'] = []
#songs['test'] = []

solos_cur.execute('SELECT * FROM solo_info') 
for row in solos_cur:
    #print(row)
    # select only songs in 4/4
    
    if row[14] == '4/4':
        song = {}
    
        song['performer'] = row[4]
        song['title'] = row[5]
        song['solo part'] = row[7] # number of solo in the original track recording
        song['instrument'] = row[8]
        song['style'] = row[9]
        song['avgtempo'] = row[10]
        
        pitch_array = []
        duration_array = []
        offset_array = []
        chord_array = []
        bass_pitch_array = []
        beat_array = []
        
        # FIX melid OF THIS SONG
        melid = row[1]
        onset = 0
        last_onset = 0
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
            
            if bar != -1:
                # SELECT ALL EVENTS IN THIS melid WITH THIS bar NUMBER AND beat NUMBER
                events_cur = con.cursor()
                events_cur.execute("SELECT * FROM melody WHERE melid = %d AND bar = %d AND beat = %d ORDER BY eventid" % (melid, bar, beat)) 
                for event_row in events_cur:
                    
                    # duration count could be adjusted to take into account
                    # of swing timing and far smaller durations!!
                    beat_duration_sec = event_row[14]
                    
                    # sampling of the measure
                    unit = beat_duration_sec * 4 / 96.
                    # possible note durations in seconds 
                    # (it is possible to add representations - include 32nds, quintuplets...):
                    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
                    possible_durations = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 72, 
                                          unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                    
                    # Define durations dictionary
                    dur_dict = {}
                    dur_dict[possible_durations[0]] = 'full'
                    dur_dict[possible_durations[1]] = 'half'
                    dur_dict[possible_durations[2]] = 'quarter'
                    dur_dict[possible_durations[3]] = '8th'
                    dur_dict[possible_durations[4]] = '16th'
                    dur_dict[possible_durations[5]] = 'dot half'
                    dur_dict[possible_durations[6]] = 'dot quarter'
                    dur_dict[possible_durations[7]] = 'dot 8th'
                    dur_dict[possible_durations[8]] = 'dot 16th'
                    dur_dict[possible_durations[9]] = 'half note triplet'
                    dur_dict[possible_durations[10]] = 'quarter note triplet'
                    dur_dict[possible_durations[11]] = '8th note triplet'
                    
                    
                    # Detect rest by onset subtraction !!!
                    # and add to arrays before notes
                    
                    # check for rests
                    onset = event_row[2]
                    intra_note_time = onset - last_onset
                    # if the interval between notes is greater than the smallest duration ('16th')
                    # and smaller than the greatest duration ('full') then there is a rest
                    if intra_note_time >= possible_durations[4]:
                        # there is a rest!
                        
                        # handle the possibility of pauses longer than a full note
                        while intra_note_time > possible_durations[0]:
                            # calculate distance from each duration
                            distance = np.abs(np.array(possible_durations) - intra_note_time)
                            idx = distance.argmin()
                            duration = dur_dict[possible_durations[idx]]
                            intra_note_time -= possible_durations[idx]
                            
                            pitch_array.append('R')
                            duration_array.append(duration)
                            chord_array.append(chord)
                            bass_pitch_array.append(bass_pitch)
                            beat_array.append(beat)
                            #velocity_array.append(velocity)                    
                            #offset_array.append(offset)
                    
                    #print(event_row)
                    pitch = event_row[3]
                    duration_sec = event_row[4]
                    
                    distance = np.abs(np.array(possible_durations) - duration_sec)
                    idx = distance.argmin()
                    duration = dur_dict[possible_durations[idx]]
                    
                    #offset = 
                    #velocity = 

                    
                    pitch_array.append(pitch)
                    duration_array.append(duration)
                    chord_array.append(chord)
                    bass_pitch_array.append(bass_pitch)
                    beat_array.append(beat)
                    #velocity_array.append(velocity)                    
                    #offset_array.append(offset)
                    last_onset = beat_duration_sec + onset

        # all these vector should have the same length
        # each element corresponds to a note event
        song['pitch'] = pitch_array
        song['duration'] = duration_array
        song['offset'] = offset_array
        song['chords'] = chord_array
        song['bass pitch'] = bass_pitch_array
        song['beats'] = beat_array
    
        # how to represent rest? 
        songs.append(song)


'''
cur = con.cursor()  
cur.execute('SELECT * FROM melody') 
for row in cur:
    print(row)
    
    
# Close connection with DB
con.close()
'''
