#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:05:49 2021

@author: vincenzomadaghiele

This scripts generates over a jazz standard given in XML format as input to the model
"""
import pretty_midi
import music21 as m21
import json
import numpy as np
import torch
import torch.nn as nn
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con

def xmlToStructuredSong(xml_path):
    # open xml file 
    # 
    #return structuredSong
    pass

def generateOverStandard(tune, model, num_chorus):
    # upload a standard in XML format
    # xml to structured song
    # use chord root as bass
    # extract chord sequence
    # generate over number of chorus chosen
    # export in midi and xml
    #return stream
    pass

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':

    # LOAD DATA
    if con.DATASET == 'WjazzDB':
            
        WjazzDB = dataset.WjazzDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                     con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
        
        #train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
        #val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getValidationData()
        #test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTestData()
        
        songs = WjazzDB.getOriginalSongDict()
        structuredSongs = WjazzDB.getStructuredSongs()
        vocabPitch, vocabDuration, vocabBeat, vocabOffset = WjazzDB.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = WjazzDB.getInverseVocabs()
        WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = WjazzDB.getChordDicts()

    elif con.DATASET == 'NottinghamDB':
        
        NottinghamDB = dataset.NottinghamDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                     con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
        
        #train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
        #val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getValidationData()
        #test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTestData()
        
        songs = NottinghamDB.getOriginalSongDict()
        structuredSongs = NottinghamDB.getStructuredSongs()
        vocabPitch, vocabDuration, vocabBeat = NottinghamDB.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix = NottinghamDB.getInverseVocabs()
        NottinghamChords, NottinghamToMusic21, NottinghamToChordComposition, NottinghamToMidiChords = NottinghamDB.getChordDicts()


    #%% LOAD PRE-TRAINED MODELS
    
    # PITCH MODEL
    isPitch = True
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 512
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 512
    
    chord_encod_dim = 64
    next_chord_encod_dim = 32

    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 64
    bass_embed_dim = 64
    
    offset_vocab_size = len(vocabOffset) # size of the duration vocabulary
    offset_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    offset_pad_idx = offset_to_ix['<pad>']
    modelPitch = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
                                      beat_vocab_size, beat_embed_dim,
                                      offset_vocab_size, offset_embed_dim,
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
                                      device, dropout, isPitch, con.COND_TYPE_PITCH).to(device)
    
    if con.DATASET == 'WjazzDB':
        savePATHpitch = 'models/MINGUSpitch_10epochs_seqLen35_WjazzDB.pt'
        
        #savePATHpitch = f'models/{con.DATASET}/pitchModel/MINGUS COND {con.COND_TYPE_PITCH} Epochs {con.EPOCHS}.pt'
        
    elif con.DATASET == 'NottinghamDB':
        savePATHpitch = 'models/MINGUSpitch_100epochs_seqLen35_NottinghamDB.pt'
    modelPitch.load_state_dict(torch.load(savePATHpitch, map_location=torch.device('cpu')))
    
    
    # DURATION MODEL
    isPitch = False
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 64
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 64
    
    chord_encod_dim = 64
    next_chord_encod_dim = 32
    
    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 32
    bass_embed_dim = 32
    
    offset_vocab_size = len(vocabOffset) # size of the duration vocabulary
    offset_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    offset_pad_idx = offset_to_ix['<pad>']
    modelDuration = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
                                      beat_vocab_size, beat_embed_dim, 
                                      offset_vocab_size, offset_embed_dim,
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
                                      device, dropout, isPitch, con.COND_TYPE_DURATION).to(device)
    
    if con.DATASET == 'WjazzDB':
        savePATHduration = 'models/MINGUSduration_10epochs_seqLen35_WjazzDB.pt'
        
        #savePATHduration = f'models/{con.DATASET}/durationModel/MINGUS COND {con.COND_TYPE_DURATION} Epochs {con.EPOCHS}.pt'
        
    elif con.DATASET == 'NottinghamDB':
        savePATHduration = 'models/MINGUSduration_100epochs_seqLen35_NottinghamDB.pt'
    modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))


    #%% Import xml file
    
    possible_durations = [4, 2, 1, 1/2, 1/4, 1/8,
                          3, 1 + 1/2, 1/2 + 1/4, 1/4 + 1/8, 
                          4/3]

    # Define durations dictionary
    dur_dict = {}
    dur_dict[possible_durations[0]] = 'full'
    dur_dict[possible_durations[1]] = 'half'
    dur_dict[possible_durations[2]] = 'quarter'
    dur_dict[possible_durations[3]] = '8th'
    dur_dict[possible_durations[4]] = '16th'
    dur_dict[possible_durations[5]] = '32th'
    dur_dict[possible_durations[6]] = 'dot half'
    dur_dict[possible_durations[7]] = 'dot quarter'
    dur_dict[possible_durations[8]] = 'dot 8th'
    dur_dict[possible_durations[9]] = 'dot 16th'
    dur_dict[possible_durations[10]] = 'half note triplet'
    inv_dur_dict = {v: k for k, v in dur_dict.items()}
    
    # invert dict from Wjazz to Music21 chords
    Music21ToWjazz = {v: k for k, v in WjazzToMusic21.items()}
    
    s = m21.converter.parse('output/xmlStandards/Billies_Bounce.xml')
    
    new_structured_song = {}
    new_structured_song['title'] = 'Billies_Bounce'
    new_structured_song['tempo'] = s.metronomeMarkBoundaries()[0][2].number
    new_structured_song['beat duration [sec]'] = 60 / new_structured_song['tempo']
    
    if not s.hasMeasures():
        s = s.makeMeasures()
    #sMeasures.show('text')
    bar_num = 0
    bars = []
    beats = []

    for measure in s.getElementsByClass('Measure'):
        bar_num += 1
        beat_num = 0
        for note in measure.notes:
            if 'ChordSymbol' in note.classSet:
                #chord
                chord = note.figure
                # if chord in Music21ToWjazz.keys():
                    # 
                # else:
                    # add to WjazzToMusic21
                    # add to WjazzToMidiChords
                    # add to WjazzToChordComposition
                print(chord)
                note.show('text')
            else:
                pitch = note.pitch.midi
                duration = note.quarterLength
                
                print(note.pitch.midi)
                print(note.quarterLength)
                
            #if note
            #note.show('text')
            #print(note.classSet)
            #print(thisNote.octave)

        #measure.show('text')
    
