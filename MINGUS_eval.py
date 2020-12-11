#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 09:20:56 2020

@author: vincenzomadaghiele

This code is done for the evaluation of my transformer model.
Metrics for evaluation:
    - MGEval (implemented, code below, some measure do not work yet):
            'total_used_pitch'
            'total_pitch_class_histogram'
            'total_used_note'
            'pitch_class_transition_matrix'
            'pitch_range'
            'avg_IOI'
            'note_length_hist'
    - Perplexity (calculated using eval function during training)
    - Cross-Entropy loss (calculated using eval function during training)
    - Accuracy (how to implement it?)
    - BLEU score (implemented, but always get zero)
    
Use tensorboard with pyTorch to obtain visualization of loss and perplexity
"""

import pretty_midi
import torch
import torch.nn as nn
import numpy as np
import math
import json
import os
import glob
from nltk.translate.bleu_score import corpus_bleu

import MINGUS_dataset_funct as dataset
import MINGUS_model as mod
import MINGUS_eval_funct as ev
import MINGUS_generate as gen

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':
    
    # DATA LOADING
    
    
    dataset_folder = "data"
    dataset_name = "w_jazz_augmented"
    pitch_path = dataset_folder +'/'+dataset_name+'/'
    
    datasetPitch = dataset.ImprovPitchDataset(pitch_path, 20)
    X_pitch = datasetPitch.getData()
    # set vocabulary for conversion
    vocabPitch = datasetPitch.vocab
    # Add padding tokens to vocab
    vocabPitch.append('<pad>')
    #vocabPitch.append('<sos>')
    #vocabPitch.append('<eos>')
    pitch_to_ix = {word: i for i, word in enumerate(vocabPitch)}
    #print(X_pitch[:3])
    
    # Divide pitch into train, validation and test
    train_pitch = X_pitch[:int(len(X_pitch)*0.7)]
    val_pitch = X_pitch[int(len(X_pitch)*0.7)+1:int(len(X_pitch)*0.7)+1+int(len(X_pitch)*0.1)]
    test_pitch = X_pitch[int(len(X_pitch)*0.7)+1+int(len(X_pitch)*0.1):]
    
    # LOAD DURATION DATASET
    #duration_path = 'data/w_jazz/'
    duration_path = dataset_folder + '/' + dataset_name + '/'
    
    datasetDuration = dataset.ImprovDurationDataset(duration_path, 10)
    X_duration = datasetDuration.getData()
    # set vocabulary for conversion
    vocabDuration = datasetDuration.vocab
    # Add padding tokens to vocab
    vocabDuration.append('<pad>')
    #vocabDuration.append('<sos>')
    #vocabDuration.append('<eos>')
    duration_to_ix = {word: i for i, word in enumerate(vocabDuration)}
    #print(X_duration[:3])
    
    # Divide duration into train, validation and test
    train_duration = X_duration[:int(len(X_duration)*0.7)]
    val_duration = X_duration[int(len(X_duration)*0.7)+1:int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1)]
    test_duration = X_duration[int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1):]
    
    
    #%% IMPORT PRE-TRAINED MODEL
    
    # HYPERPARAMETERS
    ntokens_pitch = len(vocabPitch) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    src_pad_idx = pitch_to_ix['<pad>']
    modelPitch_loaded = mod.TransformerModel(ntokens_pitch, emsize, nhead, nhid, 
                                         nlayers, src_pad_idx, device, dropout).to(device)

    # Import model
    savePATHpitch = 'models/MINGUSpitch_10epochs_seqLen35_w_jazz_augmented.pt'
    modelPitch_loaded.load_state_dict(torch.load(savePATHpitch, map_location=torch.device('cpu')))
    
    
    # HYPERPARAMETERS
    ntokens_duration = len(vocabDuration) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    src_pad_idx = pitch_to_ix['<pad>']
    modelDuration_loaded = mod.TransformerModel(ntokens_duration, emsize, nhead, nhid, 
                                            nlayers, src_pad_idx, device, dropout).to(device)

    # Import model
    savePATHduration = 'models/MINGUSduration_10epochs_seqLen35_w_jazz_augmented.pt'
    modelDuration_loaded.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))    

    
    #%% BUILD A DATASET OF GENERATED SEQUENCES
    
    generate_dataset = True
    training_path = 'data/w_jazz/*.mid'
    num_of_generations = 20
    
    if generate_dataset:
        
        standards = glob.glob(training_path)
        
        for i in range(0, num_of_generations):
            
            song_name = standards[i][12:][:-4]
            print('-'*30)
            print('Generating over song: '+ song_name)
            print('-'*30)
            
            #specify the path
            melody4gen_pitch, melody4gen_duration, dur_dict, song_properties = dataset.readMIDI(standards[i])
            melody4gen_pitch, melody4gen_duration = gen.onlyDict(melody4gen_pitch, melody4gen_duration, vocabPitch, vocabDuration)
            
            # generate entire songs given just the 40 notes
            # each generated song will have same lenght of the original song
            song_lenght= len(melody4gen_pitch) 
            melody4gen_pitch = melody4gen_pitch[:40]
            melody4gen_duration = melody4gen_duration[:40]
            
            # very high song lenght makes generation very slow
            if song_lenght > 1000:
                song_lenght = 1000
            
            notes2gen = song_lenght - 40 # number of new notes to generate
            temp = 1 # degree of randomness of the decision (creativity of the model)
            new_melody_pitch = gen.generate(modelPitch_loaded, melody4gen_pitch, 
                                        pitch_to_ix, device, next_notes=notes2gen, temperature=temp)
            new_melody_duration = gen.generate(modelDuration_loaded, melody4gen_duration, 
                                           duration_to_ix, device, next_notes=notes2gen, temperature=temp)
                    
            converted = dataset.convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)

            converted.write('output/gen4eval/'+ song_name + '_gen.mid')
        

    #%% DATA PRE-PROCESSING FOR TEST
    # this pre-processing must be the same as in the training
    
    # Maximum value of a sequence
    segment_length = 35
    train_pitch_segmented, train_duration_segmented = dataset.segmentDataset(train_pitch, train_duration, segment_length)
    val_pitch_segmented, val_duration_segmented = dataset.segmentDataset(val_pitch, val_duration, segment_length)
    test_pitch_segmented, test_duration_segmented = dataset.segmentDataset(test_pitch, test_duration, segment_length)
        
    batch_size = 20
    eval_batch_size = 10
    
    train_data_pitch = dataset.batchify(train_pitch_segmented, batch_size, pitch_to_ix, device)
    val_data_pitch = dataset.batchify(val_pitch_segmented, eval_batch_size, pitch_to_ix, device)
    test_data_pitch = dataset.batchify(test_pitch_segmented, eval_batch_size, pitch_to_ix, device)
    
    train_data_duration = dataset.batchify(train_duration_segmented, batch_size, duration_to_ix, device)
    val_data_duration = dataset.batchify(val_duration_segmented, eval_batch_size, duration_to_ix, device)
    test_data_duration = dataset.batchify(test_duration_segmented, eval_batch_size, duration_to_ix, device)
    
    # divide into target and input sequence of lenght bptt
    # --> obtain matrices of size bptt x batch_size
    bptt = segment_length # lenght of a sequence of data (IMPROVEMENT HERE!!)


    #%% BLEU score
    
    # Root directory of the generation dataset
    gen_dir = "output/gen4eval/"        
    
    #read the generated files
    files=[i for i in os.listdir(gen_dir) if i.endswith(".mid")]
    gen_pitch = np.array([np.array(dataset.readMIDI(gen_dir+i)[0], dtype=object) for i in files], dtype=object)
    gen_duration = np.array([np.array(dataset.readMIDI(gen_dir+i)[1], dtype=object) for i in files], dtype=object)
    # convert to list
    gen_ptc = [i.tolist() for i in gen_pitch]
    gen_dur = [i.tolist() for i in gen_duration]
    
    num_seq = len(gen_ptc) # number of generated sequences
    num_ref = 4 # number of reference examples for each generated sequence
    
    reference_pitch = [test_pitch[i:i+num_ref-1] for i in range(0,num_ref*num_seq,num_ref)]
    reference_ptc = [i.tolist() for i in reference_pitch]
    
    reference_duration = [test_duration[i:i+num_ref-1] for i in range(0,num_ref*num_seq,num_ref)]
    reference_dur = [i.tolist() for i in reference_duration]
    
    bleu_pitch = corpus_bleu(reference_ptc, gen_ptc)
    bleu_duration = corpus_bleu(reference_dur, gen_dur)
    
    
    #%% METRICS DICTIONARY
    
    # Instanciate dictionary
    metrics_result = {}
    metrics_result['MGEval'] = {}
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
    
    generated_path = 'output/gen4eval/*.mid'
    
    MGEresults = ev.MGEval(training_path, generated_path, num_of_generations)
    metrics_result['MGEval'] = MGEresults
    
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    testLoss_results_pitch, perplexity_results_pitch, accuracy_results_pitch  = ev.lossPerplexityAccuracy(modelPitch_loaded, test_data_pitch, 
                                                                                                          vocabPitch, criterion, bptt, device)
    metrics_result['Pitch']['Pitch_test-loss'] = testLoss_results_pitch
    metrics_result['Pitch']['Pitch_perplexity'] = perplexity_results_pitch
    metrics_result['Pitch']['Pitch_accuracy'] = accuracy_results_pitch
    metrics_result['Pitch']['Pitch_BLEU'] = bleu_pitch
    
    testLoss_results_duration, perplexity_results_duration, accuracy_results_duration  = ev.lossPerplexityAccuracy(modelDuration_loaded, test_data_duration, 
                                                                                                                   vocabDuration, criterion, bptt, device)
    metrics_result['Duration']['Duration_test-loss'] = testLoss_results_duration
    metrics_result['Duration']['Duration_perplexity'] = perplexity_results_duration
    metrics_result['Duration']['Duration_accuracy'] = accuracy_results_duration
    metrics_result['Duration']['Duration_BLEU'] = bleu_duration
    
    # Convert metrics dict to JSON and SAVE IT    
    with open('metrics/metrics_result.json', 'w') as fp:
        json.dump(metrics_result, fp)

