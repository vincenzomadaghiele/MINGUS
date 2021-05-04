#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:09:31 2021

@author: vincenzomadaghiele
"""
import glob
import json
import torch
import MINGUS_const as con
import MINGUS_condGenerate as gen
import CustomDB_dataPrep as cus
import loadDBs as dataset

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':
    
    # Load the DB to see if everything runs correctly
    
    CustomDB = dataset.CustomDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                                con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
        
    songs = CustomDB.getOriginalSongDict()
    structuredSongs = CustomDB.getStructuredSongs()
    vocabPitch, vocabDuration, vocabBeat, vocabOffset = CustomDB.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = CustomDB.getInverseVocabs()
    WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = CustomDB.getChordDicts()

    
    structuredSongs = []
    songs = []
    source_path = 'output/01_BebopNet_gens/*.xml'
    source_songs = glob.glob(source_path)
    for xml_path in source_songs:
        structuredTune = cus.xmlToStructuredSong(xml_path)
        structuredSongs.append(structuredTune)
        # export to midi
        pm = gen.structuredSongsToPM(structuredTune, WjazzToMidiChords)
        pm.write('output/01_BebopNet_gens/' + structuredTune['title'] + '.mid')

        
    # Convert dict to JSON and SAVE IT
    with open('output/01_BebopNet_gens/CustomDB_generated.json', 'w') as fp:
        json.dump(structuredSongs, fp, indent=4)
