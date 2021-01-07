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
import MINGUS_model as mod


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

