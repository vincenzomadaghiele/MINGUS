#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:44:03 2020

@author: vincenzomadaghiele
"""

from torch.utils.data import Dataset
import numpy as np
import pretty_midi
import os
from collections import Counter



# define the Pitch Dataset object for Pytorch
class ImprovPitchDataset(Dataset):
    
    def __init__(self, root_dir, freq_threshold = 5):
        
        # Root directory of the dataset
        self.root_dir = root_dir        
        
        #read all the filenames
        files=[i for i in os.listdir(self.root_dir) if i.endswith(".mid")]
        #reading each midi file
        notes_array = np.array([np.array(readMIDI(self.root_dir+i)[0], dtype=object) for i in files], dtype=object)
        
        #converting 2D array into 1D array
        notes_ = [element for note_ in notes_array for element in note_]
        unique_notes = list(set(notes_))
        print("number of unique pitches: " + str(len(unique_notes)))
        
        #computing frequency of each note
        freq = dict(Counter(notes_))
        frequent_notes = [note_ for note_, count in freq.items() if count>=freq_threshold]
        print("number of frequent pithces (more than ", freq_threshold, " times): ", (len(frequent_notes)))
        
        # prepare new musical files which contain only the top frequent notes
        new_music=[]
        for notes in notes_array:
            temp=[]
            for note_ in notes:
                if note_ in frequent_notes:
                    temp.append(note_)            
            new_music.append(temp)
        # same solos but with only most frequent notes
        new_music = np.array(new_music, dtype=object) 

        self.x = new_music
        self.num_frequent_notes = len(frequent_notes)
        self.vocab = frequent_notes

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def getData(self):
        return self.x

# define the Duration Dataset object for Pytorch
class ImprovDurationDataset(Dataset):
    
    def __init__(self, root_dir, freq_threshold = 5):
        
        # Root directory of the dataset
        self.root_dir = root_dir 
        
        #read all the filenames
        files=[i for i in os.listdir(self.root_dir) if i.endswith(".mid")]
        #reading each midi file
        notes_array = np.array([np.array(readMIDI(self.root_dir+i)[1], dtype=object) for i in files], dtype=object)
        
        #converting 2D array into 1D array
        notes_ = [element for note_ in notes_array for element in note_]
        unique_notes = list(set(notes_))
        print("number of unique durations: " + str(len(unique_notes)))
        
        #computing frequency of each note
        freq = dict(Counter(notes_))
        frequent_notes = [note_ for note_, count in freq.items() if count>=freq_threshold]
        print("number of frequent durations (more than 10 times): " + str(len(frequent_notes)))
        
        # prepare new musical files which contain only the top frequent notes
        new_music=[]
        for notes in notes_array:
            temp=[]
            for note_ in notes:
                if note_ in frequent_notes:
                    temp.append(note_)            
            new_music.append(temp)
        # same solos but with only most frequent notes
        new_music = np.array(new_music, dtype=object) 

        self.x = new_music
        self.num_frequent_notes = len(frequent_notes)
        self.vocab = frequent_notes

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def getData(self):
        return self.x

def readMIDI(file):
    '''
    
    Parameters
    ----------
    file : path to a midi file (string)
        the midi file is read by the function.

    Returns
    -------
    notes : list of notes in midi number format
        this list contains all of the notes and the rests in the song.
        The duration of each note is in the correspondent index of the durations array
    durations : list of durations in musical terms (string)
        this list contains the durations of each note and rest in the song.
        The pitch of each note is in the correspondent index of the durations array
    dur_dict : python dictionary
        the keys of this dictionary are the time in seconds 
        of each possible note/rest duration for this song.
    song_properties : python dictionary
        this dictionary contains some basic properties of the midi file.

    '''

    pm = pretty_midi.PrettyMIDI(file)
    print("Loading Music File:",file)
    
    # Get and downbeat times
    beats = pm.get_beats()
    downbeats = pm.get_downbeats()
    
    song_properties = {}
    song_properties['tempo'] = pm.estimate_tempo()
    song_properties['beat duration'] = beats[1] - beats[0]
    song_properties['measure duration'] = downbeats[2] - downbeats[0]
    
    # sampling of the measure
    unit = song_properties['measure duration'] / 96.
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
            notes.append(str(pm.instruments[instrument].notes[note].pitch))
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
    return notes, durations, dur_dict, song_properties

def getKey(val, dict_to_ix): 
    for key, value in dict_to_ix.items(): 
        if val == value: 
            return key

def convertMIDI(notes, durations, tempo, dur_dict):
    '''
    
    Parameters
    ----------
    pitches : list of pitches
        a list of all the pitches and the rests in the song to be exported.
        Each pitch/rest should have its own duration 
        at the same index of the durations list.
    durations : list of durations
        a list of all the durations of the pitches/rests in the song to be exported.
        Each duration should have its own pitch/rest 
        at the same index of the pitches list.
    tempo : integer
        tempo of the song.

    Returns
    -------
    pm : pretty_midi object
        this pretty midi can be exported to midi and saved.

    '''
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
    for i in range(min(len(notes),len(durations))):
        if notes[i] != '<pad>' and notes[i] != '<sos>' and notes[i] != '<eos>' and durations[i] != '<pad>' and durations[i] != '<sos>' and durations[i] != '<eos>':
            if notes[i] == 'R':
                duration = getKey(durations[i],dur_dict)
            else:
                pitch = int(notes[i])
                duration = getKey(durations[i],dur_dict)
                start = offset
                end = offset + duration
                inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
            offset += duration
    return pm

