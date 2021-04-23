#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:56:19 2021

@author: vincenzomadaghiele
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import collections

import loadDBs as dataset
import MINGUS_const as con

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)



if __name__ == '__main__':

    # LOAD DATA       
    WjazzDB = dataset.WjazzDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                 con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
    
    vocabPitch, vocabDuration, vocabBeat, vocabOffset = WjazzDB.getVocabs()
    
    songs = WjazzDB.getOriginalSongDict()
        
    pitch = []
    duration = []
    chord = []
    bass = []
    beat = []
    
    for song in songs['train']:
        for p in song['pitch']:
            pitch.append(p)
        for d in song['duration']:
            duration.append(d)
        for c in song['chords']:
            chord.append(c)
        for b in song['bass pitch']:
            bass.append(b)
            
    for song in songs['validation']:
        for p in song['pitch']:
            pitch.append(p)
        for d in song['duration']:
            duration.append(d)
        for c in song['chords']:
            chord.append(c)
        for b in song['bass pitch']:
            bass.append(b)
            
    for song in songs['test']:
        for p in song['pitch']:
            pitch.append(str(p))
        for d in song['duration']:
            duration.append(d)
        for c in song['chords']:
            chord.append(c)
        for b in song['bass pitch']:
            bass.append(b)    
    
    
    counter = collections.Counter(duration)
    labels, values = zip(*counter.items())
    
    indexes = np.arange(len(labels))
    width = 1
    
    plt.figure(figsize=(60, 9))
    plt.bar(indexes, values,  align='edge', width=0.3)
    plt.xticks(indexes + width * 0.5, labels, rotation=45)
    #plt.margins(0.2)
    plt.subplots_adjust(bottom = 0.2)
    plt.show()
    

    #%%
    
    styles = []
    for song in songs['train']:
        styles.append(song['style'])


    
    