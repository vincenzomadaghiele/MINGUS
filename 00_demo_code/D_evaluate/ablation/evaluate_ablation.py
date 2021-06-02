#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script evaluates the N best MINGUS model 
by conditioning on the ablation study scores
and a corpus of generated midi files
"""
import os
import sys
import argparse
import json
import math
import torch
import torch.nn as nn
import numpy as np

import B_train.loadDB as dataset
import B_train.MINGUS_model as mod
import D_evaluate.eval_funct as ev

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    
    # args for model
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
    parser.add_argument('--num_best_models', type=int, default=4,
                        help='how many models combinations to evaluate')
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
    print('number of model combinations to evaluate:', args.num_best_models)
    print('-' * 80)

    # LOAD DATA
    
    MusicDB = dataset.MusicDB(device, args.TRAIN_BATCH_SIZE, args.EVAL_BATCH_SIZE,
                 args.BPTT, args.AUGMENTATION, args.SEGMENTATION, args.augmentation_const)
    
    train_pitch_batched, train_duration_batched, train_chord_batched, train_next_chord_batched, train_bass_batched, train_beat_batched, train_offset_batched  = MusicDB.getTrainingData()
    val_pitch_batched, val_duration_batched, val_chord_batched, val_next_chord_batched, val_bass_batched, val_beat_batched, val_offset_batched  = MusicDB.getValidationData()
    test_pitch_batched, test_duration_batched, test_chord_batched, test_next_chord_batched, test_bass_batched, test_beat_batched, test_offset_batched  = MusicDB.getTestData()

    songs = MusicDB.getOriginalSongDict()
    structuredSongs = MusicDB.getStructuredSongs()
    vocabPitch, vocabDuration, vocabBeat, vocabOffset = MusicDB.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = MusicDB.getInverseVocabs()
    dbChords, dbToMusic21, dbToChordComposition, dbToMidiChords = MusicDB.getChordDicts()


    #%% Opening JSON file of multiple models scores
    
    pitch_scores = open('D_evaluate/ablation/ablation_scores/ablation_pitch.json',)
    pitch_scores = json.load(pitch_scores)
    duration_scores = open('D_evaluate/ablation/ablation_scores/ablation_duration.json',)
    duration_scores = json.load(duration_scores)
    
    NUM_MODELS = args.num_best_models
    # find best conditionings by accuracy
    pitch_conds = []
    pitch_acc = []
    for cond, metrics in pitch_scores.items():
        pitch_conds.append(cond)
        pitch_acc.append(metrics['val acc'])
    
    pitch_acc = np.array(pitch_acc)
    pitch_conds = np.array(pitch_conds)

    sorted_index_pitch = np.argsort(pitch_acc)
    sorted_acc_pitch = pitch_acc[sorted_index_pitch]
    BEST_CONDS_PITCH = pitch_conds[sorted_index_pitch][::-1][:NUM_MODELS]
    
    duration_conds = []
    duration_acc = []
    for cond, metrics in duration_scores.items():
        duration_conds.append(cond)
        duration_acc.append(metrics['val acc'])
    
    duration_acc = np.array(duration_acc)
    duration_conds = np.array(duration_conds)

    sorted_index_duration = np.argsort(duration_acc)
    sorted_acc_duration = duration_acc[sorted_index_duration]
    BEST_CONDS_DURATION = duration_conds[sorted_index_duration][::-1][:NUM_MODELS]

    for COND_PITCH in BEST_CONDS_PITCH:
        for COND_DURATION in BEST_CONDS_DURATION:
            # LOAD PRE-TRAINED MODELS
            
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
                                              device, dropout, isPitch, COND_PITCH).to(device)
            
            savePATHpitch = f'B_train/models/pitchModel/MINGUS COND {COND_PITCH} Epochs {args.EPOCHS}.pt'
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
                                              device, dropout, isPitch, COND_DURATION).to(device)
            
            savePATHduration = f'B_train/models/durationModel/MINGUS COND {COND_DURATION} Epochs {args.EPOCHS}.pt'
            modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))
        
        
            #%% GENERATE MUSIC WITH THIS MODELS COMBINATION
            
            cmd = f'"python3 C_generate/generate.py --COND_TYPE_PITCH {COND_PITCH} --COND_TYPE_DURATION {COND_DURATION} --GENERATE_CORPUS --EPOCHS {args.EPOCHS}"'
            os.system('cmd /k '+cmd+'')
            
            
            #%% Load generated structured songs
            
            path = 'A_preprocessData/data/DATA.json'
            with open(path) as f:
                original_structuredSongs = json.load(f)
            original_structuredSongs = original_structuredSongs['structured for generation']
            
            path = 'C_generate/generated_tunes/generated.json'
            with open(path) as f:
                generated_structuredSongs = json.load(f)
                
            
            #%% METRICS DICTIONARY
            
            # Make directory to save metrics file
            parent_directory = 'D_evaluate/metrics/'
            model_id = f'MINGUS PITCH_COND {COND_PITCH} DUR_COND {COND_DURATION} Epochs {args.EPOCHS}'
            path = os.path.join(parent_directory, model_id + '/')
            if not os.path.isdir(path):
                os.mkdir(path)
            
            # Instanciate dictionary
            metrics_result = {}
            metrics_result['MGEval'] = {}
            metrics_result['Harmonic coherence'] = {}
            metrics_result['Pitch'] = {}
            metrics_result['Duration'] = {}
            metrics_result['Pitch']['Pitch_accuracy'] = {}
            metrics_result['Pitch']['Pitch_perplexity'] = {}
            metrics_result['Pitch']['Pitch_test-loss'] = {}
            metrics_result['Pitch']['Pitch_BLEU'] = {}
            metrics_result['Duration']['Duration_accuracy'] = {}
            metrics_result['Duration']['Duration_perplexity'] = {}
            metrics_result['Duration']['Duration_test-loss'] = {}
            metrics_result['Duration']['Duration_BLEU'] = {}
            
            # MGEval on generated midi files
            num_of_generations = 14
            original_path = 'D_evaluate/reference4eval/*.mid'
            generated_path = 'C_generate/generated_tunes/*.mid'
            MGEresults = ev.MGEval(original_path, generated_path, path, num_of_generations)
            metrics_result['MGEval'] = MGEresults
            
            # Harmonic coherence of generated samples
            original_scale_coherence, original_chord_coherence = ev.HarmonicCoherence(original_structuredSongs, 
                                                                                      dbToMusic21, 
                                                                                      dbToMidiChords)
            generated_scale_coherence, generated_chord_coherence = ev.HarmonicCoherence(generated_structuredSongs, 
                                                                                        dbToMusic21, 
                                                                                        dbToMidiChords)
            
            metrics_result['Harmonic coherence']['Original scale coherence'] = np.round_(original_scale_coherence, decimals=4)
            metrics_result['Harmonic coherence']['Original chord coherence'] = np.round_(original_chord_coherence, decimals=4)
            metrics_result['Harmonic coherence']['Generated scale coherence'] = np.round_(generated_scale_coherence, decimals=4)
            metrics_result['Harmonic coherence']['Generated chord coherence'] = np.round_(generated_chord_coherence, decimals=4)
            
            
            # loss, perplexity and accuracy of pitch model
            isPitch = True
            criterion = nn.CrossEntropyLoss(ignore_index=pitch_pad_idx)    
            testLoss_results_pitch, accuracy_results_pitch = mod.evaluate(modelPitch, pitch_to_ix, 
                                                                        test_pitch_batched, test_duration_batched, 
                                                                        test_chord_batched, test_next_chord_batched,
                                                                        test_bass_batched, test_beat_batched, test_offset_batched,
                                                                        criterion, args.BPTT, device, isPitch)
            
            # BLEU score
            bleu_pitch, bleu_duration = ev.BLEUscore(original_structuredSongs, generated_structuredSongs)
            
            metrics_result['Pitch']['Pitch_test-loss'] = np.round_(testLoss_results_pitch, decimals=4)
            metrics_result['Pitch']['Pitch_perplexity'] = np.round_(math.exp(testLoss_results_pitch), decimals=4)
            metrics_result['Pitch']['Pitch_accuracy'] = np.round_(accuracy_results_pitch * 100, decimals=4)
            metrics_result['Pitch']['Pitch_BLEU'] = np.round_(bleu_pitch, decimals=4)
            
            # loss, perplexity and accuracy of duration model
            isPitch = False
            criterion = nn.CrossEntropyLoss(ignore_index=duration_pad_idx)
            testLoss_results_duration, accuracy_results_duration = mod.evaluate(modelDuration, duration_to_ix, 
                                                                            test_pitch_batched, test_duration_batched, 
                                                                            test_chord_batched, test_next_chord_batched,
                                                                            test_bass_batched, test_beat_batched, test_offset_batched,
                                                                            criterion, args.BPTT, device, isPitch)
            
            metrics_result['Duration']['Duration_test-loss'] = np.round_(testLoss_results_duration, decimals=4)
            metrics_result['Duration']['Duration_perplexity'] = np.round_(math.exp(testLoss_results_duration), decimals=4)
            metrics_result['Duration']['Duration_accuracy'] = np.round_(accuracy_results_duration * 100, decimals=4)
            metrics_result['Duration']['Duration_BLEU'] = np.round_(bleu_duration, decimals=4)
            
            # Convert metrics dict to JSON and SAVE IT
            with open(path + 'metrics' + model_id + '.json', 'w') as fp:
                json.dump(metrics_result, fp, indent=4)
            
        
        
        
