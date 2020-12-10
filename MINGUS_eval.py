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
import MINGUS_dataset_funct as dataset
import MINGUS_model as mod
import MINGUS_eval_funct as ev
import MINGUS_generate as gen

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


if __name__ == '__main__':
    
    # DATA LOADING
    
    
    # LOAD PITCH DATASET
    pitch_path = 'data/w_jazz/'
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
    duration_path = 'data/w_jazz/'
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
    savePATHpitch = 'modelsPitch/modelPitch_10epochs_wjazz_segmented.pt'
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
    savePATHduration = 'modelsDuration/modelDuration_10epochs_wjazz_segmented.pt'
    modelDuration_loaded.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))    

    
    #%% BUILD A DATASET OF GENERATED SEQUENCES
    
    training_path = 'data/w_jazz/*.mid'
    
    import glob
    standards = glob.glob(training_path)
    num_of_generations = 100
    j=0
    # for BLEU score
    #candidate_corpus_pitch = []
    #candidate_corpus_duration = []
    #references_corpus_pitch = []
    #references_corpus_duration = []
    for i in range(0, num_of_generations):
        
        # update reference corpus for BLEU
        #references_corpus_pitch.append(melody4gen_pitch[:60])
        #references_corpus_duration.append(melody4gen_duration[:60])

        # update candidate corpus for BLEU
        #candidate_corpus_pitch.append(new_melody_pitch[:60])
        #candidate_corpus_duration.append(new_melody_pitch[:60])
        
        #specify the path
        melody4gen_pitch, melody4gen_duration, dur_dict, song_properties = dataset.readMIDI(standards[i])
        melody4gen_pitch, melody4gen_duration = gen.onlyDict(melody4gen_pitch, melody4gen_duration, vocabPitch, vocabDuration)
        melody4gen_pitch = melody4gen_pitch[:40]
        melody4gen_duration = melody4gen_duration[:40]
        #print(melody4gen_pitch)
        #print(melody4gen_duration)
        
        notes2gen = 20 # number of new notes to generate
        temp = 1 # degree of randomness of the decision (creativity of the model)
        new_melody_pitch = gen.generate(modelPitch_loaded, melody4gen_pitch, 
                                    pitch_to_ix, device, next_notes=notes2gen, temperature=temp)
        new_melody_duration = gen.generate(modelDuration_loaded, melody4gen_duration, 
                                       duration_to_ix, device, next_notes=notes2gen, temperature=temp)
        
        print('length of gen melody: ', len(new_melody_pitch))
        print('generated pitches: ', np.array(new_melody_pitch[40:]) )
    
        converted = dataset.convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
        
        song_name = standards[i][12:][:-4]
        print('-'*30)
        print('Generating over song: '+ song_name)
        print('-'*30)
        #converted.write('output/gen4eval/music'+str(j)+'.mid')
        converted.write('output/gen4eval/'+ song_name + '_gen.mid')
        
        j+=1


    #%% Melody Segmentation
    
    # Maximum value of a sequence
    segment_length = 35
    train_pitch_segmented, train_duration_segmented = dataset.segmentDataset(train_pitch, train_duration, segment_length)
    val_pitch_segmented, val_duration_segmented = dataset.segmentDataset(val_pitch, val_duration, segment_length)
    test_pitch_segmented, test_duration_segmented = dataset.segmentDataset(test_pitch, test_duration, segment_length)


    #%% Perplexity, Test Loss, Accuracy
    
    #DATA PREPARATION FOR TEST
    
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

    
    #%% METRICS DICTIONARY
    
    # Instanciate dictionary
    metrics_result = {}
    metrics_result['MGEval'] = {}
    metrics_result['BLEU'] = {}
    metrics_result['Pitch_accuracy'] = {}
    metrics_result['Pitch_perplexity'] = {}
    metrics_result['Pitch_test-loss'] = {}
    metrics_result['Duration_accuracy'] = {}
    metrics_result['Duration_perplexity'] = {}
    metrics_result['Duration_test-loss'] = {}
    
    
    generated_path = 'output/gen4eval/*.mid'
    
    MGEresults = ev.MGEval(training_path, generated_path, num_of_generations)
    metrics_result['MGEval'] = MGEresults
    
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    perplexity_results_pitch, testLoss_results_pitch, accuracy_results_pitch  = ev.lossPerplexityAccuracy(modelPitch_loaded, test_data_pitch, vocabPitch, criterion, bptt, device)
    metrics_result['Pitch_perplexity'] = perplexity_results_pitch
    metrics_result['Pitch_test-loss'] = testLoss_results_pitch
    metrics_result['Pitch_accuracy'] = accuracy_results_pitch
    
    perplexity_results_duration, testLoss_results_duration, accuracy_results_duration  = ev.lossPerplexityAccuracy(modelDuration_loaded, test_data_duration, vocabDuration, criterion, bptt, device)
    metrics_result['Duration_perplexity'] = perplexity_results_duration
    metrics_result['Duration_test-loss'] = testLoss_results_duration
    metrics_result['Duration_accuracy'] = accuracy_results_duration
    
    # Convert metrics dict to JSON and SAVE IT    
    with open('metrics/metrics_result.json', 'w') as fp:
        json.dump(metrics_result, fp)
    
    
    #%% MODEL EVALUATION
    # accuracy, perplexity (paper seq-Attn)
    # NLL loss, BLEU (paper explicitly conditioned melody generation)
    
    #parameters = []
    #for param in modelDuration_loaded.parameters():
        #parameters.append(param.data.numpy())
        #print(param.data)
    
    """
    BLEU code: it always gets 0, don't understand why
    
    def formatBLEU(corpus, max_n=1):
        new_corpus = []
        for song in corpus:
            for i in range(0, len(song), max_n):
                new_corpus.append(song[i:i+max_n])
        return new_corpus
    max_n = 2
    weights=[0.5, 0.5]
    candidate_corpus_pitch_reformat = formatBLEU(candidate_corpus_pitch, max_n)
    references_corpus_pitch_reformat = formatBLEU(references_corpus_pitch, max_n)
    from torchtext.data.metrics import bleu_score
    pitchBLEU = bleu_score(candidate_corpus_pitch, references_corpus_pitch, max_n = max_n, weights = weights)
    #durationBLEU = bleu_score(candidate_corpus_duration, references_corpus_duration, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    """
    
    