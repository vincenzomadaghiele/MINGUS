#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:39:06 2021

@author: vincenzomadaghiele
for info about the note_seq : https://github.com/magenta/note-seq/blob/master/note_seq/protobuf/music.proto#L252
"""

import json
import numpy as np
import music21 as m21
import glob
from note_seq import abc_parser
import note_seq


if __name__=="__main__":
    
    '''
    path = 'data/nottingham-dataset-master/ABC_cleaned/'
    abc = '*.abc'
    abcFiles = glob.glob(path + abc)
    for file in abcFiles:
        abcScore = m21.converter.parse(file)
    '''
    
    path = 'data/nottingham-dataset-master/ABC_cleaned/ashover.abc'
    #stream = m21.abcFormat.translate.abcToStreamOpus(path)
    abcSongbook = abc_parser.parse_abc_tunebook_file(path)
    #for abcSong in abcSongbook[0]:
    note_seq.plot_sequence(abcSongbook[0][8])
    #for tempo in abcSongbook[0][8].tempos:
    
    for timesignature in abcSongbook[0][8].time_signatures:
        print(timesignature.numerator)
    
    for textannotation in abcSongbook[0][8].text_annotations:
        print(textannotation.text)
        print(textannotation.annotation_type) # if 1 = chord
        print(textannotation.time) # time of the annotation
        
    metadata = abcSongbook[0][8].sequence_metadata
    print(metadata.title)
    print(metadata.artist)
    print(metadata.genre)
    print(metadata.composers) 
        

    
    #print(abcSongbook[0][8].total_time)
    #for note in abcSongbook[0][8].notes:
        #print(note)
    #note_seq.sequence_proto_to_midi_file(abcSongbook[0][8], 'output/abcExp.mid')