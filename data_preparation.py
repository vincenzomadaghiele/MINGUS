#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:56:44 2021

@author: vincenzomadaghiele
"""
import sqlite3

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
        song['beat duration'] = 0
        
        pitch_array = []
        duration_array = []
        offset_array = []
        chord_array = []
        bass_pitch_array = []
        
        # FIX melid OF THIS SONG
        melid = row[1]
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
                    
                    # Detect rest by onset subtraction !!!
                    
                    print(event_row)
                    pitch = event_row[3]
                    beat_duration_sec = event_row[14]
                    duration_sec = event_row[4]
                    duration = 0
                    
                    # also velocity could be extracted
                    
                break

        # all these vector should have the same length
        # each element corresponds to a note event
        song['pitch'] = []
        song['duration'] = []
        song['offset'] = []
        song['chords'] = []
        song['bass pitch'] = []
        
        break
    
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
