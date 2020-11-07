#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 08:56:16 2020

@author: vincenzomadaghiele
@name: pretty_midi_demo
from: https://nbviewer.jupyter.org/github/craffel/pretty-midi/blob/master/Tutorial.ipynb
"""

import pretty_midi
import numpy as np
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt
#%matplotlib inline
# For putting audio in the notebook
import IPython.display



# We'll load in the example.mid file distributed with pretty_midi
pm = pretty_midi.PrettyMIDI('data/w_jazz/ArtPepper_Anthropology_FINAL.mid')

# Function to plot midi on piano roll
def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

plt.figure(figsize=(12, 4))
plot_piano_roll(pm, 0, 127)

# EXTRACT INFORMATION
# Let's look at what's in this MIDI file
print('There are {} time signature changes'.format(len(pm.time_signature_changes)))
print('There are {} instruments'.format(len(pm.instruments)))
print('Instrument 1 has {} notes'.format(len(pm.instruments[0].notes)))

# What's the start time of the 10th note on the 3rd instrument?
print(pm.instruments[0].notes[10].start)
# What's that in ticks?
tick = pm.time_to_tick(pm.instruments[0].notes[10].start)
print(tick)
# Note we can also go in the opposite direction
print(pm.tick_to_time(int(tick)))


# ADJUST THE TIMING
# Get the length of the MIDI file
length = pm.get_end_time()
# This will effectively slow it down to 110% of its original length
pm.adjust_times([0, length], [0, length*1.1])
# Let's check what time our tick from above got mapped to - should be 1.1x
print(pm.tick_to_time(tick))


# Get and downbeat times
beats = pm.get_beats()
downbeats = pm.get_downbeats()
# Plot piano roll
plt.figure(figsize=(12, 4))
plot_piano_roll(pm, 24, 84)
ymin, ymax = plt.ylim()
# Plot beats as grey lines, downbeats as white lines
mir_eval.display.events(beats, base=ymin, height=ymax, color='#AAAAAA')
mir_eval.display.events(downbeats, base=ymin, height=ymax, color='#FFFFFF', lw=2)
# Only display 20 seconds for clarity
plt.xlim(25, 45);
plt.figure()

# Plot a pitch class distribution - sort of a proxy for key
plt.bar(np.arange(12), pm.get_pitch_class_histogram());
plt.xticks(np.arange(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.xlabel('Note')
plt.ylabel('Proportion')
plt.figure()


#%% PRETTY MIDI ON A W_JAZZ SOLO

file = 'data/w_jazz/ArtPepper_Anthropology_FINAL.mid'
print("Loading Music File:",file)
pm = pretty_midi.PrettyMIDI(file)

tempo = pm.estimate_tempo()
# Get and downbeat times
beats = pm.get_beats()
downbeats = pm.get_downbeats()
beat_duration = beats[1] - beats[0]
measure_duration = downbeats[2] - downbeats[0]

# sampling of the measure
unit = measure_duration / 96.
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

# compile the lists of pitchs and durations
notes = []
durations = []
for instrument in range(len(pm.instruments)):
    for note in range(len(pm.instruments[instrument].notes)-1):
        # append pitch
        notes.append(pm.instruments[instrument].notes[note].pitch)
        # calculate note duration in secods
        duration_sec = pm.instruments[instrument].notes[note].end - pm.instruments[instrument].notes[note].start
        # calculate distance from each duration
        distance = np.abs(np.array(possible_durations) - duration_sec)
        idx = distance.argmin()
        durations.append(dur_dict[possible_durations[idx]])
        
        # check for rests
        intra_note_time = pm.instruments[instrument].notes[note+1].start - pm.instruments[instrument].notes[note].end
        # if the interval between notes is greater than the smallest duration ('16th')
        # and smaller than the greatest duration ('full') then there is a rest
        if intra_note_time >= possible_durations[4]:
            # there is a rest!
            
            # handle the possibility of pauses longer than a full note
            while intra_note_time > possible_durations[0]:
                notes.append('R')
                # calculate distance from each duration
                distance = np.abs(np.array(possible_durations) - intra_note_time)
                idx = distance.argmin()
                durations.append(dur_dict[possible_durations[idx]])
                intra_note_time -= possible_durations[idx]
            
            notes.append('R')
            # calculate distance from each duration
            distance = np.abs(np.array(possible_durations) - intra_note_time)
            idx = distance.argmin()
            durations.append(dur_dict[possible_durations[idx]])
            
        

def getKey(val, dict_to_ix): 
    for key, value in dict_to_ix.items(): 
        if val == value: 
            return key

    
def convertMIDI(pitches, durations, tempo):
    # pitches and durations must be of equal lenght
    # still does not include rests
    
    # Construct a PrettyMIDI object.
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Add a piano instrument
    inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    pm.instruments.append(inst)
    # Let's add a few notes to our instrument
    velocity = 100
    offset = 0
    # for each note
    for i in range(len(notes)):
        if notes[i] == 'R':
            duration = getKey(durations[i],dur_dict)
        else:
            pitch = notes[i]
            duration = getKey(durations[i],dur_dict)
            start = offset
            end = offset + duration
            inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
        offset += duration
    return pm


converted = convertMIDI(notes, durations, tempo)
converted.write('output/out.mid')
    
        


"""
#defining function to read MIDI files
def read_midi_pitch(file):
    
    print("Loading Music File:",file)
    
    notes=[]
    notes_to_parse = None
    
    #parsing a midi file
    midi = converter.parse(file)
    
    #grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    #Looping over all the instruments
    for part in s2.parts:
        notes_to_parse = part.recurse() 
        #finding whether a particular element is note or a rest
        for element in notes_to_parse:        
            #note
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, note.Rest):
                notes.append('R')
            #chord
            #elif isinstance(element, chord.Chord):
                #notes.append('.'.join(str(n) for n in element.normalOrder))
                
    return np.array(notes)
"""