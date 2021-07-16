#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains the functions for MINGUS
evaluation with different metrics
"""
import music21 as m21
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

def MGEval(training_midi_path, generated_midi_path, fig_savePath, num_samples = 20):
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
    set1_eval['avg_IOI'] = np.zeros((num_samples,1))
    set1_eval['avg_pitch_shift'] = np.zeros((num_samples,1))
    set1_eval['note_length_hist'] = np.zeros((num_samples,12))
    set1_eval['total_pitch_class_histogram'] = np.zeros((num_samples,12))
    set1_eval['note_length_transition_matrix'] = np.zeros((num_samples,12,12))
    set1_eval['pitch_class_transition_matrix'] = np.zeros((num_samples,12,12))
    set1_eval['pitch_range'] = np.zeros((num_samples,1))
    set1_eval['total_used_note'] = np.zeros((num_samples,1))
    set1_eval['total_used_pitch'] = np.zeros((num_samples,1))
    
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
    set2_eval['avg_IOI'] = np.zeros((num_samples,1))
    set2_eval['avg_pitch_shift'] = np.zeros((num_samples,1))
    set2_eval['note_length_hist'] = np.zeros((num_samples,12))
    set2_eval['total_pitch_class_histogram'] = np.zeros((num_samples,12))
    set2_eval['note_length_transition_matrix'] = np.zeros((num_samples,12,12))
    set2_eval['pitch_class_transition_matrix'] = np.zeros((num_samples,12,12))
    set2_eval['pitch_range'] = np.zeros((num_samples,1))
    set2_eval['total_used_note'] = np.zeros((num_samples,1))
    set2_eval['total_used_pitch'] = np.zeros((num_samples,1))
    
    # Calculate metrics
    for j in range(0, len(metrics_list)):
        for i in range(0, num_samples):
            feature = core.extract_feature(set2[i])
            set2_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature)
    
    # Print the results
    for i in range(0, len(metrics_list)):
        
        # mean and std of the reference set
        results[metrics_list[i]]['ref_mean'] = np.round_(np.mean(set1_eval[metrics_list[i]]).tolist(), decimals=4)
        results[metrics_list[i]]['ref_std'] = np.round_(np.std(set1_eval[metrics_list[i]]).tolist(), decimals=4)
        # mean and std of the generated set
        results[metrics_list[i]]['gen_mean'] = np.round_(np.mean(set2_eval[metrics_list[i]]).tolist(), decimals=4)
        results[metrics_list[i]]['gen_std'] = np.round_(np.std(set2_eval[metrics_list[i]]).tolist(), decimals=4)
        
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
        #print(plot_set1_intra[i])
        fig = plt.figure()
        sns.kdeplot(plot_set1_intra[i], label='intra_set1')
        sns.kdeplot(plot_sets_inter[i], label='inter')
        sns.kdeplot(plot_set2_intra[i], label='intra_set2')
    
        plt.title(metrics_list[i])
        plt.xlabel('Euclidean distance')
        plt.show()
        fig.savefig(fig_savePath + metrics_list[i] + '.png')
        
    # Calculate divergence between measures
    for i in range(0, len(metrics_list)):
        
        # mean and std of the reference set
        results[metrics_list[i]]['ref_KL-div'] = np.round_(utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i]), decimals=4)
        results[metrics_list[i]]['ref_overlap-area'] = np.round_(utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i]), decimals=4)
        # mean and std of the generated set
        results[metrics_list[i]]['gen_KL-div'] = np.round_(utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i]), decimals=4)
        results[metrics_list[i]]['gen_overlap-area'] = np.round_(utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i]), decimals=4)
            
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


def BLEUscore(original_structuredSongs, generated_structuredSongs):
        
    # get arrays from structured songs
    original_tunes_pitch = []
    original_tunes_duration = []
    for tune in original_structuredSongs:
        newtune_pitch = []
        newtune_duration = []
        for bar in tune['bars']:
            for beat in bar['beats']:
                for pitch in beat['pitch']:
                    newtune_pitch.append(pitch)
                for duration in beat['duration']:
                    newtune_duration.append(duration)
        original_tunes_pitch.append(newtune_pitch)
        original_tunes_duration.append(newtune_duration)
        
    generated_tunes_pitch = []
    generated_tunes_duration = []
    for tune in original_structuredSongs:
        newtune_pitch = []
        newtune_duration = []
        for bar in tune['bars']:
            for beat in bar['beats']:
                for pitch in beat['pitch']:
                    newtune_pitch.append(pitch)
                for duration in beat['duration']:
                    newtune_duration.append(duration)
        generated_tunes_pitch.append(newtune_pitch)
        generated_tunes_duration.append(newtune_duration)
    
    num_seq = len(generated_tunes_pitch) # number of generated sequences
    num_ref = 4 # number of reference examples for each generated sequence
    
    reference_pitch = [original_tunes_pitch[i:i+num_ref-1] for i in range(0,num_ref*num_seq,num_ref)]
    reference_duration = [original_tunes_duration[i:i+num_ref-1] for i in range(0,num_ref*num_seq,num_ref)]
    
    bleu_pitch = corpus_bleu(reference_pitch[:4], generated_tunes_pitch[:4])
    bleu_duration = corpus_bleu(reference_duration[:4], generated_tunes_duration[:4])
    
    return bleu_pitch, bleu_duration

def HarmonicCoherence(structuredSongs, chordToMusic21, datasetToMidiChord):
    
    print('Evaluating harmonic coherence')
    print('Computing scales...')
    
    datasetChordToScale = {}
    for chord in datasetToMidiChord.keys():
        if chord != 'NC':
            m21chord = chordToMusic21[chord]
            try: 
                h = m21.harmony.ChordSymbol(m21chord)
                hd = m21.harmony.ChordStepModification('add', 2)
                h.addChordStepModification(hd, updatePitches=True)
                hd = m21.harmony.ChordStepModification('add', 3)
                h.addChordStepModification(hd, updatePitches=True)
                hd = m21.harmony.ChordStepModification('add', 4)
                h.addChordStepModification(hd, updatePitches=True)
                hd = m21.harmony.ChordStepModification('add', 5)
                h.addChordStepModification(hd, updatePitches=True)
                hd = m21.harmony.ChordStepModification('add', 6)
                h.addChordStepModification(hd, updatePitches=True)
                hd = m21.harmony.ChordStepModification('add', 7)
                h.addChordStepModification(hd, updatePitches=True)
                hd = m21.harmony.ChordStepModification('add', 8)
                h.addChordStepModification(hd, updatePitches=True)
                scale = [m21.pitch.Pitch(pitch).name for pitch in h.pitches]
                datasetChordToScale[chord] = scale
                #print(chord + ': ' + scale)
            except:
                print('Could not convert chord', m21chord)
    
    print('Evaluating songs...')
    scale_coherence = 0
    chord_coherence = 0
    count_pitch = 0
    for tune in structuredSongs:
        for bar in tune['bars']:
            for beat in bar['beats']:
                chord = beat['chord']
                if chord != 'NC' and chord in datasetToMidiChord.keys() and chord in datasetChordToScale.keys():
                    # derive chord scale
                    scale = datasetChordToScale[chord]
                    
                    # derive chord pitch
                    midiChord = datasetToMidiChord[chord]
                    chordPitch = [m21.pitch.Pitch(midiPitch).name for midiPitch in midiChord if midiPitch != 'R']
                    
                    for pitch in beat['pitch']:
                        if pitch != 'R':
                            pitchName = m21.pitch.Pitch(pitch).name
                            if pitchName in scale:
                                scale_coherence += 1
                            if pitchName in chordPitch:
                                chord_coherence += 1
                            count_pitch += 1
    
    scale_coherence = scale_coherence / count_pitch
    chord_coherence = chord_coherence / count_pitch
    return scale_coherence, chord_coherence
