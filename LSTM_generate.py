#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 08:58:57 2021

@author: vincenzomadaghiele
"""
import pretty_midi
import torch
import torch.nn as nn
import numpy as np
import math
import json
import MINGUS_dataset_funct as dataset
import LSTM_model as mod


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

def getNote(val, dict_to_ix): 
    for key, value in dict_to_ix.items(): 
         if val == value: 
             return key

def generate(model, melody4gen, dict_to_ix, device, bptt=35, next_notes=10, temperature = 1):
    '''
    
    Parameters
    ----------
    model : pytorch Model
        Model to be used for generation.
    melody4gen : numpy ndarray
        melody to be used as a generation starting point.
    dict_to_ix : python dictionary
        dictionary used for model training.
    bptt : integer, optional
        standard lenght of the sequence used for training. The default is 35.
    next_notes : integer, optional
        Number of notes to be generated. The default is 10.

   Returns
   -------
   melody4gen_list : list
       original melody with generated notes appended at the end.
    
    '''
    model.eval()
    melody4gen_list = melody4gen.tolist()
    with torch.no_grad():
        for i in range(0,next_notes):
            # prepare input to the model
            melody4gen_batch = batch4gen(np.array(melody4gen_list), len(melody4gen_list), dict_to_ix, device)
            
            # reshape to column vector
            melody4gen_batch = melody4gen_batch.reshape(melody4gen_batch.shape[1], melody4gen_batch.shape[0])
            #melody4gen_batch = melody4gen_batch.view(1,-1)
            #print(melody4gen_batch.shape)
            model.init_hidden_and_cell(melody4gen_batch.shape[1])
            y_pred = model(melody4gen_batch)
            
            #word_weights = y_pred[-1].squeeze().div(temperature).exp().cpu()
            #word_idx = torch.multinomial(word_weights, 1)[0]
            
            word_idx = int(torch.exp(y_pred[-1, :, :]).multinomial(1))
            #word_idx = int(torch.exp(y_pred[:, -1, :]).multinomial(1))
            
            #last_note_logits = y_pred[-1,-1]
            #_, max_idx = torch.max(last_note_logits, dim=0)
            melody4gen_list.append(getNote(word_idx, dict_to_ix))
    
    return melody4gen_list


# Remove characters who are not in the dictionary
def onlyDict(pitchs, durations, vocabPitch, vocabDuration):
    '''

    Parameters
    ----------
    pitchs : numpy ndarray
        array of pitch of the melody.
    durations : numpy ndarray
        array of duration of the melody.
    vocabPitch : python dictionary
        dictionary used for pitch training.
    vocabDuration : python dictionary
        dictionary used for duration training.

    Returns
    -------
    new_pitch : numpy ndarray
        pitch of the melody. 
        The ones who were not in the dictionary have been removed.
    new_duration : numpy ndarray
        duration of the melody. 
        The ones who were not in the dictionary have been removed.

    '''
    new_pitch = []
    new_duration = []
    for i in range(len(pitchs)):
        if pitchs[i] in vocabPitch and durations[i] in vocabDuration:
            new_pitch.append(pitchs[i]) 
            new_duration.append(durations[i]) 
    new_pitch = np.array(new_pitch) # same solos but with only most frequent notes
    new_duration = np.array(new_duration) # same solos but with only most frequent notes
    return new_pitch, new_duration
    
# This is used in the generate() function
def batch4gen(data, bsz, dict_to_ix, device):
    '''

    Parameters
    ----------
    data : numpy ndarray or list
        data to be batched.
    bsz : integer
        batch size for generation.
    dict_to_ix : python dictionary
        dictionary used for model training.

    Returns
    -------
    pytorch Tensor
        tensor of data tokenized and batched for generation.

    '''
        
    #padded = pad(data)
    padded_num = [dict_to_ix[x] for x in data]
    
    data = torch.tensor(padded_num, dtype=torch.long)
    data = data.contiguous()
    
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
    
def generateEqual(model, melody4gen, dict_to_ix, device, bptt=35, next_notes=10):
    '''
    
    Parameters
    ----------
    model : pytorch Model
        Model to be used for generation.
    melody4gen : numpy ndarray
        melody to be used as a generation starting point.
    dict_to_ix : python dictionary
        dictionary used for model training.
    bptt : integer, optional
        standard lenght of the sequence used for training. The default is 35.
    next_notes : integer, optional
        Number of notes to be generated. The default is 10.
    
    Returns
    -------
    melody4gen_list : list
        original melody with generated notes appended at the end.
    
    '''
    model.eval()
    melody4gen_list = melody4gen.tolist()
    new_melody = []
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
            
        # prepare input to the model
        melody4gen_batch = batch4gen(np.array(melody4gen_list), len(melody4gen_list), dict_to_ix, device)
            
        # reshape to column vector
        melody4gen_batch = melody4gen_batch.reshape(melody4gen_batch.shape[1], melody4gen_batch.shape[0])
            
        if melody4gen_batch.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(melody4gen_batch.size(0)).to(device)
    
        y_pred = model(melody4gen_batch, src_mask)
        #print(y_pred.size(0))
        for j in range(y_pred.size(0)):
            note_logits = y_pred[j,0,:]
            _, max_idx = torch.max(note_logits, dim=0)
            new_melody.append(getNote(max_idx, dict_to_ix))

            
        ac = 0
        for k in range(1,len(new_melody)):
            #print(new_melody[k])
            #print(melody4gen_list[k-1])
            if new_melody[k] == melody4gen_list[k-1]:
                ac += 1
        print("Accuracy", ac,'/',len(melody4gen_list))
            
            # last_note_logits = y_pred[-1,-1,:]
            #_, max_idx = torch.max(last_note_logits, dim=0)
            #melody4gen_list.append(getNote(max_idx, dict_to_ix))
    
    return new_melody


if __name__ == '__main__':
    
    # DATA LOADING
    
    
    # LOAD PITCH DATASET
    pitch_path = 'data/folkDB/'
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
    duration_path = 'data/folkDB/'
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
    src_pad_idx = pitch_to_ix['<pad>']

    # These values are set as the default used in the ECMG paper
    vocab_size = len(vocabPitch)
    embed_dim = 8 # 8 for pitch, 4 for duration
    # the dimension of the output is the same 
    # as the input because the vocab is the same
    output_dim = len(vocabPitch)
    hidden_dim = 256
    seq_len = 32
    num_layers = 2
    batch_size = 32
    dropout = 0.5
    batch_norm = False # in training it is true
    no_cuda = False

    modelPitch_loaded = mod.NoCondLSTM(vocab_size, embed_dim, output_dim, 
                                hidden_dim, seq_len, num_layers, batch_size, 
                                dropout, batch_norm, no_cuda).to(device)
    
    # Import model
    savePATHpitch = 'models/LSTMpitch_10epochs_seqLen32_w_jazz.pt'
    modelPitch_loaded.load_state_dict(torch.load(savePATHpitch, map_location=torch.device('cpu')))
    
    
    # HYPERPARAMETERS
    src_pad_idx = pitch_to_ix['<pad>']

    # These values are set as the default used in the ECMG paper
    vocab_size = len(vocabDuration)
    embed_dim = 4 # 8 for pitch, 4 for duration
    # the dimension of the output is the same 
    # as the input because the vocab is the same
    output_dim = len(vocabDuration)
    hidden_dim = 256
    seq_len = 32
    num_layers = 2
    batch_size = 32
    dropout = 0.5
    batch_norm = False # in training it is true
    no_cuda = False

    modelDuration_loaded = mod.NoCondLSTM(vocab_size, embed_dim, output_dim, 
                                hidden_dim, seq_len, num_layers, batch_size, 
                                dropout, batch_norm, no_cuda).to(device)

    # Import model
    savePATHduration = 'models/LSTMduration_10epochs_seqLen32_w_jazz.pt'
    modelDuration_loaded.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))    
   
    #%% ONE SONG GENERATION WITH PRE-TRAINED MODELS
    '''
    bptt = 35
    #specify the path
    f = 'data/w_jazz/JohnColtrane_Mr.P.C._FINAL.mid'
    melody4gen_pitch, melody4gen_duration, dur_dict, song_properties = dataset.readMIDI(f)
    melody4gen_pitch, melody4gen_duration = onlyDict(melody4gen_pitch, melody4gen_duration, vocabPitch, vocabDuration)
    melody4gen_pitch = melody4gen_pitch[:80]
    melody4gen_duration = melody4gen_duration[:80]
    
    notes2gen = 40 # number of new notes to generate
    temp = 1 # degree of randomness of the decision (creativity of the model)
    new_melody_pitch = generate(modelPitch_loaded, melody4gen_pitch, pitch_to_ix, device,
                                next_notes=notes2gen, temperature=temp)
    new_melody_duration = generate(modelDuration_loaded, melody4gen_duration, duration_to_ix, device,
                                   next_notes=notes2gen, temperature=temp)
    
    # convert to midi
    converted = dataset.convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
    converted.write('output/generated_music.mid')
    
    new_melody_pitch = generateEqual(modelPitch_loaded, melody4gen_pitch, pitch_to_ix, device, next_notes=notes2gen)
    new_melody_duration = generateEqual(modelDuration_loaded, melody4gen_duration, duration_to_ix, device, next_notes=notes2gen)
    
    # convert to midi
    converted = dataset.convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
    converted.write('output/equal.mid')
    '''
    
    #%% BUILD A DATASET OF GENERATED SEQUENCES
    
    # Set to True to generate dataset of songs
    generate_dataset = False
    
    if generate_dataset:
        training_path = 'data/folkDB/*.mid'
        
        import glob
        standards = glob.glob(training_path)
        num_of_generations = 20

        for i in range(0, num_of_generations):
            
            song_name = standards[i][12:][:-4]
            print('-'*30)
            print('Generating over song: '+ song_name)
            print('-'*30)
            
            #specify the path
            melody4gen_pitch, melody4gen_duration, dur_dict, song_properties = dataset.readMIDI(standards[i])
            melody4gen_pitch, melody4gen_duration = onlyDict(melody4gen_pitch, melody4gen_duration, vocabPitch, vocabDuration)
            
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
            new_melody_pitch = generate(modelPitch_loaded, melody4gen_pitch, 
                                        pitch_to_ix, device, next_notes=notes2gen, temperature=temp)
            new_melody_duration = generate(modelDuration_loaded, melody4gen_duration, 
                                           duration_to_ix, device, next_notes=notes2gen, temperature=temp)
            
            print('length of gen melody: ', len(new_melody_pitch))
        
            converted = dataset.convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
            converted.write('output/gen4eval_folkDB/'+ song_name + '_genLSTM.mid')

