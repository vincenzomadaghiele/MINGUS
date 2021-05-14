#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:00:14 2021

@author: vincenzomadaghiele
"""

import sys
import argparse
import glob
import json
import torch

#import A_preprocessData.data_preprocessing as cus
import B_train.loadDB as dataset
import B_train.MINGUS_model as mod
import C_generate.gen_funct as gen

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    
    # args for model
    parser.add_argument('--COND_TYPE_PITCH', type=str, default='I-C-NC-B-BE-O',
                    help='conditioning features for pitch model')
    parser.add_argument('--COND_TYPE_DURATION', type=str, default='I-C-NC-B-BE-O',
                    help='conditioning features for duration model')
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=20,
                        help='training batch size')
    parser.add_argument('--EVAL_BATCH_SIZE', type=int, default=10,
                        help='evaluation batch size')
    parser.add_argument('--BPTT', type=int, default=35,
                    help='length of a note sequence for training')
    parser.add_argument('--EPOCHS', type=int, default=10,
                    help='epochs for training')
    parser.add_argument('--SEGMENTATION', action='store_false', default=True,
                        help='augment dataset')
    parser.add_argument('--AUGMENTATION', action='store_true', default=False,
                        help='augment dataset')
    parser.add_argument('--augmentation_const', type=int, default=3,
                        help='how many times to augment the data')
    # args for generation
    parser.add_argument('--NUM_CHORUS', type=int, default=3,
                        help='number of chorus of the improvisation')
    parser.add_argument('--TEMPERATURE', type=int, default=1,
                        help='amount of randomness of the generation [0,1]')
    parser.add_argument('--xmlSTANDARD', type=str, default='C_generate/xml4gen/All_The_Things_You_Are_short.xml',
                        help='path to the xml standard to generate notes on')
    parser.add_argument('--GENERATE_CORPUS', action='store_true', default=False,
                        help='create a corpus of generated tunes')
    args = parser.parse_args(sys.argv[1:])
    
    # Constants for MINGUS training
    print('Model summary:')
    print('-' * 80)
    print('TRAIN_BATCH_SIZE:', args.TRAIN_BATCH_SIZE)
    print('EVAL_BATCH_SIZE:', args.EVAL_BATCH_SIZE)
    print('EPOCHS:', args.EPOCHS)
    print('sequence length:', args.BPTT)
    print('SEGMENTATION:', args.SEGMENTATION)
    print('AUGMENTATION:', args.AUGMENTATION) 
    print('augmentation_const:', args.augmentation_const)
    print('pitch model conditionings:', args.COND_TYPE_PITCH)
    print('duration model conditionings:', args.COND_TYPE_DURATION)
    print('-' * 80)
    print('Generation summary:')
    print('NUM_CHORUS:', args.NUM_CHORUS)
    print('TEMPERATURE:', args.TEMPERATURE)
    print('xmlSTANDARD:', args.xmlSTANDARD)
    print('GENERATE_CORPUS:', args.GENERATE_CORPUS)
    print('-' * 80)

    # LOAD DATA
    
    MusicDB = dataset.MusicDB(device, args.TRAIN_BATCH_SIZE, args.EVAL_BATCH_SIZE,
                 args.BPTT, args.AUGMENTATION, args.SEGMENTATION, args.augmentation_const)
        
    #train_pitch_batched, train_duration_batched, train_chord_batched, train_next_chord_batched, train_bass_batched, train_beat_batched, train_offset_batched  = MusicDB.getTrainingData()
    #val_pitch_batched, val_duration_batched, val_chord_batched, val_next_chord_batched, val_bass_batched, val_beat_batched, val_offset_batched  = MusicDB.getValidationData()
    #test_pitch_batched, test_duration_batched, test_chord_batched, test_next_chord_batched, test_bass_batched, test_beat_batched, test_offset_batched  = MusicDB.getTestData()

    songs = MusicDB.getOriginalSongDict()
    structuredSongs = MusicDB.getStructuredSongs()
    vocabPitch, vocabDuration, vocabBeat, vocabOffset = MusicDB.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = MusicDB.getInverseVocabs()
    dbChords, dbToMusic21, dbToChordComposition, dbToMidiChords = MusicDB.getChordDicts()


    #%% LOAD PRE-TRAINED MODELS
    
    # PITCH MODEL
    # ensure that these parameters are the same of the trained models
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
                                      device, dropout, isPitch, args.COND_TYPE_PITCH).to(device)
    
    savePATHpitch = f'B_train/models/pitchModel/MINGUS COND {args.COND_TYPE_PITCH} Epochs {args.EPOCHS}.pt'
    modelPitch.load_state_dict(torch.load(savePATHpitch, map_location=torch.device('cpu')))
    
    
    # DURATION MODEL
    # ensure that these parameters are the same of the trained models
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
                                      device, dropout, isPitch, args.COND_TYPE_DURATION).to(device)
    
    savePATHduration = f'B_train/models/durationModel/MINGUS COND {args.COND_TYPE_DURATION} Epochs {args.EPOCHS}.pt'
    modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))


    #%% Import xml file
    
    #xml_path = 'C_generate/xml4gen/All_The_Things_You_Are_short.xml'    
    tuneFromXML, WjazzToMusic21, WjazzToMidiChords, WjazzToChordComposition, WjazzChords = gen.xmlToStructuredSong(args.xmlSTANDARD, 
                                                                                                               dbToMusic21,
                                                                                                               dbToMidiChords, 
                                                                                                               dbToChordComposition, 
                                                                                                               dbChords)
    
    #tuneFromXML = cus.xmlToStructuredSong(xml_path)

    # GENERATE ON A TUNE    
    isJazz = False
    new_structured_song = gen.generateOverStandard(tuneFromXML, args.NUM_CHORUS, args.TEMPERATURE, 
                                               modelPitch, modelDuration, dbToMidiChords, 
                                               pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                               vocabPitch, vocabDuration,
                                               args.BPTT, device,
                                               isJazz)
    title = new_structured_song['title']
    pm = gen.structuredSongsToPM(new_structured_song, dbToMidiChords, isJazz)
    pm.write('C_generate/generated_tunes/' + title + '.mid')


    #%% BUILD A DATASET OF GENERATED TUNES
    
    # Set to True to generate dataset of songs
    if args.GENERATE_CORPUS:
        generated_structuredSongs = []
        original_structuredSongs = []
        source_path = 'C_generate/xml4gen/*.xml'
        source_songs = glob.glob(source_path)
        for xml_path in source_songs:
            
            isJazz = False
            tune, WjazzToMusic21, WjazzToMidiChords, WjazzToChordComposition, WjazzChords = gen.xmlToStructuredSong(xml_path, 
                                                                                                               WjazzToMusic21,
                                                                                                               WjazzToMidiChords, 
                                                                                                               WjazzToChordComposition, 
                                                                                                               WjazzChords)

            new_structured_song = gen.generateOverStandard(tune, args.NUM_CHORUS, args.TEMPERATURE, 
                                                   modelPitch, modelDuration, WjazzToMidiChords, 
                                                   pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                                   vocabPitch, vocabDuration,
                                                   args.BPTT, device,
                                                   isJazz)
            
            pm = gen.structuredSongsToPM(new_structured_song, WjazzToMidiChords, isJazz)
            pm.write('C_generate/generated_tunes/' + new_structured_song['title'] + '.mid')
            generated_structuredSongs.append(new_structured_song)
            
        # convert reference to midi and structured song json
        source_path = 'D_evaluate/reference4eval/xml/*.xml'
        source_songs = glob.glob(source_path)
        reference_structuredSongs = []
        for xml_path in source_songs:
            #tune = cus.xmlToStructuredSong(xml_path) # remove min_rest
            tune, WjazzToMusic21, WjazzToMidiChords, WjazzToChordComposition, WjazzChords = gen.xmlToStructuredSong(xml_path, 
                                                                                                                   WjazzToMusic21,
                                                                                                                   WjazzToMidiChords, 
                                                                                                                   WjazzToChordComposition, 
                                                                                                                   WjazzChords)

            reference_structuredSongs.append(tune)
            pm = gen.structuredSongsToPM(tune, WjazzToMidiChords)
            pm.write('D_evaluate/reference4eval/' + tune['title'] + '.mid')
        
        # write structured songs to json for evaluation
        with open('C_generate/generated_tunes/generated.json', 'w') as fp:
            json.dump(generated_structuredSongs, fp, indent=4)
        with open('D_evaluate/reference4eval/reference.json', 'w') as fp:
            json.dump(reference_structuredSongs, fp, indent=4)



