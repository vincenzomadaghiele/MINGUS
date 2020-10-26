#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:44:03 2020

@author: vincenzomadaghiele
"""

# library for understanding music
from music21 import *
from torch.utils.data import Dataset
import numpy as np

# define the Duration Dataset object for Pytorch
class ImprovDurationDataset(Dataset):
    
    """
    --> DataLoader can do the batch computation for us

    Implement a custom Dataset:
    inherit Dataset
    implement __init__ , __getitem__ , and __len__
    """
    
    def __init__(self):
        #for listing down the file names
        import os
        
        #specify the path
        path='data/w_jazz/'
        #read all the filenames
        files=[i for i in os.listdir(path) if i.endswith(".mid")]
        #reading each midi file
        notes_array = np.array([read_midi_duration(path+i) for i in files])
        
        #converting 2D array into 1D array
        notes_ = [element for note_ in notes_array for element in note_]
        #No. of unique notes
        unique_notes = list(set(notes_))
        print("number of unique notes: " + str(len(unique_notes)))
        
        from collections import Counter
        #computing frequency of each note
        freq = dict(Counter(notes_))
              
        # the threshold for frequent notes can change 
        threshold=50 # this threshold is the number of classes that have to be predicted
        frequent_notes = [note_ for note_, count in freq.items() if count>=threshold]
        print("number of frequent notes (more than 50 times): " + str(len(frequent_notes)))
        self.num_frequent_notes = len(frequent_notes)
        self.vocab = frequent_notes
        
        # prepare new musical files which contain only the top frequent notes
        new_music=[]
        for notes in notes_array:
            temp=[]
            for note_ in notes:
                if note_ in frequent_notes:
                    temp.append(note_)            
            new_music.append(temp)
        new_music = np.array(new_music) # same solos but with only most frequent notes

        self.x = new_music

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def getData(self):
        return self.x

#defining function to read MIDI files duration
def read_midi_duration(file):
    
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
        #finding whether a particular element is note or a chord
        for element in notes_to_parse:        
            #note
            if isinstance(element, note.Note):
                # Read only duration of the note
                notes.append((element.duration.quarterLength))        
            #chord
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    return np.array(notes)

# define the Pitch Dataset object for Pytorch
class ImprovPitchDataset(Dataset):
    
    """
    --> DataLoader can do the batch computation for us

    Implement a custom Dataset:
    inherit Dataset
    implement __init__ , __getitem__ , and __len__
    """
    
    def __init__(self):
        #for listing down the file names
        import os
        
        #specify the path
        path='data/w_jazz/'
        #read all the filenames
        files=[i for i in os.listdir(path) if i.endswith(".mid")]
        #reading each midi file
        notes_array = np.array([read_midi_pitch(path+i) for i in files])
        
        #converting 2D array into 1D array
        notes_ = [element for note_ in notes_array for element in note_]
        #No. of unique notes
        unique_notes = list(set(notes_))
        print("number of unique notes: " + str(len(unique_notes)))
        
        from collections import Counter
        #computing frequency of each note
        freq = dict(Counter(notes_))
              
        # the threshold for frequent notes can change 
        threshold=50 # this threshold is the number of classes that have to be predicted
        frequent_notes = [note_ for note_, count in freq.items() if count>=threshold]
        print("number of frequent notes (more than 50 times): " + str(len(frequent_notes)))
        self.num_frequent_notes = len(frequent_notes)
        self.vocab = frequent_notes
        
        # prepare new musical files which contain only the top frequent notes
        new_music=[]
        for notes in notes_array:
            temp=[]
            for note_ in notes:
                if note_ in frequent_notes:
                    temp.append(note_)            
            new_music.append(temp)
        new_music = np.array(new_music) # same solos but with only most frequent notes

        self.x = new_music

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def getData(self):
        return self.x

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
        #finding whether a particular element is note or a chord
        for element in notes_to_parse:        
            #note
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))        
            #chord
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    return np.array(notes)

# Convert the output of the model to midi
def convert_to_midi(new_melody_pitch, new_melody_duration):
    offset = 0
    output_notes = stream.Stream()
    # create note and chord objects based on the values generated by the model
    for i in range(len(new_melody_pitch)):
        new_note = note.Note()
        new_note.pitch.nameWithOctave = new_melody_pitch[i]
        new_note.duration.quarterLength = float(new_melody_duration[i])
        #new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
        
        # increase offset each iteration so that notes do not stack
        offset += 1

    #midi_stream = stream.Stream(output_notes)
    output_notes.write('midi', fp='output/music.mid')  

