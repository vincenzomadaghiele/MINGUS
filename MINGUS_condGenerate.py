#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 09:01:54 2021

@author: vincenzomadaghiele
"""

import torch
import torch.nn as nn
import math
import time
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':

    # LOAD DATA 
    
    WjazzDB = dataset.WjazzDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                 con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
    
    #train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
    #val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getValidationData()
    #test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTestData()
    
    songs = WjazzDB.getOriginalSongDict()
    vocabPitch, vocabDuration, vocabBeat = WjazzDB.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix = WjazzDB.getInverseVocabs()
    WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = WjazzDB.getChordDicts()

    #%% Conversion to MIDI

    trialSong = songs['train'][0]