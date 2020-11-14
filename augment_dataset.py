#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:59:52 2020

@author: vincenzomadaghiele
"""

import pretty_midi
import glob


def transposeFile(file, transpose_value = 8, transposeUp = True):
    '''

    Parameters
    ----------
    file : string
        path of the midi file to be transposed.
    transpose_value : int, optional
        how many semitones to transpose the midi file. The default is 8.
    transposeUp : boolean, optional
        True = transpose up, False = transpose down. 
        The default is True.

    Returns
    -------
    transposed : pretty_midi object
        transposed midi file to be saved.

    '''
    
    pm = pretty_midi.PrettyMIDI(file)
    tempo = pm.estimate_tempo()
    
    # set up transpose value
    transposed = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for instrument in range(len(pm.instruments)):
        # Add a piano instrument
        inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
        transposed.instruments.append(inst)
        for note in range(len(pm.instruments[instrument].notes)):
            # transpose each note
            if transposeUp == True:
                new_pitch = pm.instruments[instrument].notes[note].pitch + transpose_value
            else:
                new_pitch = pm.instruments[instrument].notes[note].pitch - transpose_value
            velocity = pm.instruments[instrument].notes[note].velocity
            start = pm.instruments[instrument].notes[note].start
            end = pm.instruments[instrument].notes[note].end
            # add transposed notes to the new track
            transposed.instruments[instrument].notes.append(pretty_midi.Note(velocity, new_pitch, start, end))
    
    return transposed

# make +-4
if __name__ == '__main__':
    # Initialize dataset1 (training data)
    source_path = 'data/w_jazz/*.mid'
    transpose_path = 'data/w_jazz_augmented/'
    source_songs = glob.glob(source_path)
    
    #transpose_value = 8 # number of semitones to transpose
    for transpose_value in range (1,5):
        for i in range(len(source_songs)):
            song_name = source_songs[i][12:-4]
            transposedUp = transposeFile(source_songs[i], transpose_value, transposeUp = True)
            # save file 
            transposedUp.write(transpose_path + song_name + 'transposedUpBy' + str(transpose_value) + '.mid')
            
            transposedDown = transposeFile(source_songs[i], transpose_value, transposeUp = False)
            # save file 
            transposedDown.write(transpose_path + song_name + 'transposedDownBy' + str(transpose_value) + '.mid')
    

