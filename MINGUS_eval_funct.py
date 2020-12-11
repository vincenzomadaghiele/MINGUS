#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:35:11 2020

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


def lossPerplexityAccuracy(eval_model, data_source, vocab, criterion, bptt, device):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(vocab)
    # accuracy is calculated token-by-token
    tot_tokens = data_source.shape[0]*data_source.shape[1]
    correct = 0
    src_mask = eval_model.generate_square_subsequent_mask(bptt).to(device)
        
    #tot_tokens = 0 # REMOVE
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            # get batch
            data, targets, targets_no_reshape = mod.get_batch(data_source, i, bptt)
            if data.size(0) != bptt:
                src_mask = eval_model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            # calculate loss function
            total_loss += len(data) * criterion(output_flat, targets).item()
                
            # get accuracy for each element of the batch (very inefficent)
            for k in range(output.shape[1]):
                word_logits = output[:,k]
                for j in range(len(word_logits)):
                    logit = word_logits[j]
                    _, max_idx = torch.max(logit, dim=0)
                    
                    # exclude pad tokens for pitch and duration
                    #if targets_no_reshape[j,k] != 49 and targets_no_reshape[j,k] != 12: # REMOVE
                    correct += (max_idx == targets_no_reshape[j,k]).sum().item()
                    #tot_tokens += 1 # REMOVE
        
    accuracy = correct / tot_tokens *100
    loss = total_loss / (len(data_source) - 1)
    perplexity = math.exp(loss)
    return loss, perplexity, accuracy


def MGEval(training_midi_path, generated_midi_path, num_samples = 20):
    '''

    Parameters
    ----------
    training_midi_path : TYPE
        DESCRIPTION.
    generated_midi_path : TYPE
        DESCRIPTION.
    num_samples : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    '''
    
    # Build a dataset of generated sequences (coherent with training data)
    # How to build the dataset of generated sequences? 
    # How many tokens for each sequence? 
    # With how many samples to compare?
    # ---------
    # Calculate MGEval metrics for the two datasets and compute KL-divergence
        
    import glob
    import seaborn as sns
    import matplotlib.pyplot as plt
    from mgeval import core, utils
    from sklearn.model_selection import LeaveOneOut
        
    # Initalize results dictionary
    results = {}
    
    # Initialize dataset1 (training data)
    set1 = glob.glob(training_midi_path)
    # Dictionary of metrics
    set1_eval = {'total_used_pitch':np.zeros((num_samples,1))}
    # Add metrics to the dictionary
    set1_eval['total_pitch_class_histogram'] = np.zeros((num_samples,12))
    #set1_eval['total_used_note'] = np.zeros((num_samples,1))
    set1_eval['pitch_class_transition_matrix'] = np.zeros((num_samples,12,12))
    #set1_eval['note_length_transition_matrix'] = np.zeros((num_samples,12,12))
    set1_eval['pitch_range'] = np.zeros((num_samples,1))
    #set1_eval['avg_pitch_shift'] = np.zeros((num_samples,1))
    set1_eval['avg_IOI'] = np.zeros((num_samples,1))
    #set1_eval['note_length_hist'] = np.zeros((num_samples,12))
    
    # Calculate metrics
    metrics_list = list(set1_eval.keys())
    for metric in metrics_list:
        results[metric] = {}
    
    for j in range(0, len(metrics_list)):
        for i in range(0, num_samples):
            feature = core.extract_feature(set1[i])
            set1_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature)
    
    
    # Initialize dataset2 (generated samples)
    set2 = glob.glob(generated_midi_path)
    # Dictionary of metrics
    set2_eval = {'total_used_pitch':np.zeros((num_samples,1))}
    # Add metrics to the dictionary
    set2_eval['total_pitch_class_histogram'] = np.zeros((num_samples,12))
    #set2_eval['total_used_note'] = np.zeros((num_samples,1))
    set2_eval['pitch_class_transition_matrix'] = np.zeros((num_samples,12,12))
    #set2_eval['note_length_transition_matrix'] = np.zeros((num_samples,12,12)) # problem: note enough generation make the mean 0
    set2_eval['pitch_range'] = np.zeros((num_samples,1))
    #set2_eval['avg_pitch_shift'] = np.zeros((num_samples,1))
    set2_eval['avg_IOI'] = np.zeros((num_samples,1))
    #set2_eval['note_length_hist'] = np.zeros((num_samples,12))
    
    # Calculate metrics
    for j in range(0, len(metrics_list)):
        for i in range(0, num_samples):
            feature = core.extract_feature(set2[i])
            set2_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature)
        
    # Print the results
    for i in range(0, len(metrics_list)):
            
        # mean and std of the reference set
        results[metrics_list[i]]['ref_mean'] = np.mean(set1_eval[metrics_list[i]]).tolist()
        results[metrics_list[i]]['ref_std'] = np.std(set1_eval[metrics_list[i]]).tolist()
        # mean and std of the generated set
        results[metrics_list[i]]['gen_mean'] = np.mean(set2_eval[metrics_list[i]]).tolist()
        results[metrics_list[i]]['gen_std'] = np.std(set2_eval[metrics_list[i]]).tolist()
            
        # print the results
        print( metrics_list[i] + ':')
        print('------------------------')
        print(' Reference set')
        print('  mean: ', np.mean(set1_eval[metrics_list[i]]))
        print('  std: ', np.std(set1_eval[metrics_list[i]]))
    
        print('------------------------')
        print(' Generated set')
        print('  mean: ', np.mean(set2_eval[metrics_list[i]]))
        print('  std: ', np.std(set2_eval[metrics_list[i]]))
            
        print()
        
    # exhaustive cross-validation for intra-set distances measurement
    loo = LeaveOneOut() # compare each sequence with all the others in the set
    loo.get_n_splits(np.arange(num_samples))
    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    j = 0
    for metric in metrics_list:
        for train_index, test_index in loo.split(np.arange(num_samples)):
            # compute the distance between each song in the datasets 
            # and all the others in the same dataset
            set1_intra[test_index[0]][j] = utils.c_dist(set1_eval[metric][test_index], set1_eval[metric][train_index])
            set2_intra[test_index[0]][j] = utils.c_dist(set2_eval[metric][test_index], set2_eval[metric][train_index])
        j += 1
        
        
        # exhaustive cross-validation for inter-set distances measurement
    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
    j = 0
    for metric in metrics_list:
        for train_index, test_index in loo.split(np.arange(num_samples)):
            # compute the distance between each song in the training dataset 
            # and all the samples in the generated dataset
            sets_inter[test_index[0]][j] = utils.c_dist(set1_eval[metric][test_index], set2_eval[metric])
        j += 1
        
    # Plotting the results
    plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
    plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
    plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)
    for i in range(0,len(metrics_list)):
        sns.kdeplot(plot_set1_intra[i], label='intra_set1')
        sns.kdeplot(plot_sets_inter[i], label='inter')
        sns.kdeplot(plot_set2_intra[i], label='intra_set2')
    
        plt.title(metrics_list[i])
        plt.xlabel('Euclidean distance')
        plt.show()
        
    # Calculate divergence between measures
    for i in range(0, len(metrics_list)):
        
        # mean and std of the reference set
        results[metrics_list[i]]['ref_KL-div'] = utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
        results[metrics_list[i]]['ref_overlap-area'] = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
        # mean and std of the generated set
        results[metrics_list[i]]['gen_KL-div'] = utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])
        results[metrics_list[i]]['gen_overlap-area'] = utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])
            
        print(metrics_list[i] + ':')
        print('------------------------')
        print(' Reference set')
        print('  Kullback–Leibler divergence:',utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i]))
        print('  Overlap area:', utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i]))
        
        print(' Generated set')
        print('  Kullback–Leibler divergence:',utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i]))
        print('  Overlap area:', utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i]))
        print()
        
    return results
