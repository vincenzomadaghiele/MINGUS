#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:46:41 2020

@author: vincenzomadaghiele
"""

import midi
import glob
import numpy as np
import pretty_midi
import seaborn as sns
import matplotlib.pyplot as plt
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut

# MGEval compares two different datasets (training and generated dataset) 
# This code is a demo comparing two parts of the training data

#%% Absolute measurement: statistic analysis

# Num of samples for each dataset
num_samples = 20


# Initialize dataset1 (training data)
set1 = glob.glob('../data/w_jazz_mini/*.mid')
# Dictionary of metrics
set1_eval = {'total_used_pitch':np.zeros((num_samples,1))}
# Add metrics to the dictionary
set1_eval['total_used_note'] = np.zeros((num_samples,1))
set1_eval['total_pitch_class_histogram'] = np.zeros((num_samples,12))
set1_eval['note_length_hist'] = np.zeros((num_samples,12))
set1_eval['pitch_class_transition_matrix'] = np.zeros((num_samples,12,12))
set1_eval['note_length_transition_matrix'] = np.zeros((num_samples,12,12))
set1_eval['pitch_range'] = np.zeros((num_samples,1))
set1_eval['avg_pitch_shift'] = np.zeros((num_samples,1))
set1_eval['avg_IOI'] = np.zeros((num_samples,1))


metrics_list = list(set1_eval.keys())
for j in range(0, len(metrics_list)):
    for i in range(0, num_samples):
        feature = core.extract_feature(set1[i])
        set1_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature)


# Initialize dataset2 (generated samples)
set2 = glob.glob('../data/w_jazz_mini2/*.mid')
# Dictionary of metrics
set2_eval = {'total_used_pitch':np.zeros((num_samples,1))}
# Add metrics to the dictionary
set2_eval['total_used_note'] = np.zeros((num_samples,1))
set2_eval['total_pitch_class_histogram'] = np.zeros((num_samples,12))
set2_eval['note_length_hist'] = np.zeros((num_samples,12))
set2_eval['pitch_class_transition_matrix'] = np.zeros((num_samples,12,12))
set2_eval['note_length_transition_matrix'] = np.zeros((num_samples,12,12))
set2_eval['pitch_range'] = np.zeros((num_samples,1))
set2_eval['avg_pitch_shift'] = np.zeros((num_samples,1))
set2_eval['avg_IOI'] = np.zeros((num_samples,1))


for j in range(0, len(metrics_list)):
    for i in range(0, num_samples):
        feature = core.extract_feature(set2[i])
        set2_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature)
    
# Print the results
for i in range(0, len(metrics_list)):
    print( metrics_list[i] + ':')
    print('------------------------')
    print(' demo_set')
    print('  mean: ', np.mean(set1_eval[metrics_list[i]], axis=0))
    print('  std: ', np.std(set1_eval[metrics_list[i]], axis=0))

    print('------------------------')
    print(' demo_set')
    print('  mean: ', np.mean(set2_eval[metrics_list[i]], axis=0))
    print('  std: ', np.std(set2_eval[metrics_list[i]], axis=0))
    
    print()
    

#%% Relative measurement: generalizes the result among features with various dimensions

# Adjust for other metrics !!!!

# exhaustive cross-validation for intra-set distances measurement
loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
j = 0
for metric in metrics_list:
    for train_index, test_index in loo.split(np.arange(num_samples)):
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
    print(metrics_list[i] + ':')
    print('------------------------')
    print(' demo_set1')
    print('  Kullback–Leibler divergence:',utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i]))
    print('  Overlap area:', utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i]))
    
    print(' demo_set2')
    print('  Kullback–Leibler divergence:',utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i]))
    print('  Overlap area:', utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i]))
    print()


