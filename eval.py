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
Understand if BLEU and accuracy are necessary
Probably need to implement some export function 
of the different metrics on a .csv or .txt file
"""

import pretty_midi
import torch
import torch.nn as nn
import numpy as np
import math
import json
from MINGUS_dataset_funct import ImprovDurationDataset, ImprovPitchDataset, readMIDI, convertMIDI

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

# TRANSFORMER MODEL
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, src_pad_idx, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.src_pad_idx = src_pad_idx
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_src_pad_mask(self, src):
        pad_mask = src.transpose(0, 1) == self.src_pad_idx
        #pad_mask = pad_mask.float().masked_fill(pad_mask == True, float('-inf')).masked_fill(pad_mask == False, float(0.0))
        return pad_mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src_padding_mask = self.make_src_pad_mask(src)
        #print(src_padding_mask)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_padding_mask.to(device))
        #print(output)
        #output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def getNote(val, dict_to_ix): 
    for key, value in dict_to_ix.items(): 
         if val == value: 
             return key

def generate(model, melody4gen, dict_to_ix, bptt=35, next_notes=10):
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
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0,next_notes):
            # prepare input to the model
            melody4gen_batch = batch4gen(np.array(melody4gen_list), len(melody4gen_list), dict_to_ix)
            
            # reshape to column vector
            melody4gen_batch = melody4gen_batch.reshape(melody4gen_batch.shape[1], melody4gen_batch.shape[0])
            
            if melody4gen_batch.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(melody4gen_batch.size(0)).to(device)

            y_pred = model(melody4gen_batch, src_mask)
            last_note_logits = y_pred[-1,-1]
            _, max_idx = torch.max(last_note_logits, dim=0)
            melody4gen_list.append(getNote(max_idx, dict_to_ix))

    return melody4gen_list


if __name__ == '__main__':
    
    # DATA LOADING
    
    
    # LOAD PITCH DATASET
    pitch_path = 'data/w_jazz/'
    datasetPitch = ImprovPitchDataset(pitch_path, 20)
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
    datasetDuration = ImprovDurationDataset(duration_path, 10)
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
    modelPitch_loaded = TransformerModel(ntokens_pitch, emsize, nhead, nhid, nlayers, dropout).to(device)

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
    modelDuration_loaded = TransformerModel(ntokens_duration, emsize, nhead, nhid, nlayers, dropout).to(device)

    # Import model
    savePATHduration = 'modelsDuration/modelDuration_10epochs_wjazz_segmented.pt'
    modelDuration_loaded.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))

    
    #%% ONE SONG GENERATION WITH PRE-TRAINED MODELS

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
    def batch4gen(data, bsz, dict_to_ix):
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
    
    def generateEqual(model, melody4gen, dict_to_ix, bptt=35, next_notes=10):
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
            melody4gen_batch = batch4gen(np.array(melody4gen_list), len(melody4gen_list), dict_to_ix)
            
            # reshape to column vector
            melody4gen_batch = melody4gen_batch.reshape(melody4gen_batch.shape[1], melody4gen_batch.shape[0])
            print(melody4gen_batch.shape)
            
            if melody4gen_batch.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(melody4gen_batch.size(0)).to(device)
    
            y_pred = model(melody4gen_batch, src_mask)
            #print(y_pred.size(0))
            for j in range(y_pred.size(0)):
                note_logits = y_pred[j,0,:]
                _, max_idx = torch.max(note_logits, dim=0)
                #print(max_idx)
                new_melody.append(getNote(max_idx, dict_to_ix))
            
            #print(new_melody)
            #print(melody4gen_list)
            
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
    
    bptt = 50
    #specify the path
    f = 'data/w_jazz/JohnColtrane_Mr.P.C._FINAL.mid'
    melody4gen_pitch, melody4gen_duration, dur_dict, song_properties = readMIDI(f)
    melody4gen_pitch, melody4gen_duration = onlyDict(melody4gen_pitch, melody4gen_duration, vocabPitch, vocabDuration)
    melody4gen_pitch = melody4gen_pitch[:80]
    melody4gen_duration = melody4gen_duration[:80]
    
    notes2gen = 40 # number of new notes to generate
    new_melody_pitch = generate(modelPitch_loaded, melody4gen_pitch, pitch_to_ix, next_notes=notes2gen)
    new_melody_duration = generate(modelDuration_loaded, melody4gen_duration, duration_to_ix, next_notes=notes2gen)
    
    # convert to midi
    converted = convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
    converted.write('output/generated_music.mid')
    
    new_melody_pitch = generateEqual(modelPitch_loaded, melody4gen_pitch, pitch_to_ix, next_notes=notes2gen)
    new_melody_duration = generateEqual(modelDuration_loaded, melody4gen_duration, duration_to_ix, next_notes=notes2gen)
    
    # convert to midi
    converted = convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
    converted.write('output/equal.mid')
    
    
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
        melody4gen_pitch, melody4gen_duration, dur_dict, song_properties = readMIDI(standards[i])
        melody4gen_pitch, melody4gen_duration = onlyDict(melody4gen_pitch, melody4gen_duration, vocabPitch, vocabDuration)
        melody4gen_pitch = melody4gen_pitch[:40]
        melody4gen_duration = melody4gen_duration[:40]
        #print(melody4gen_pitch)
        #print(melody4gen_duration)
        
        notes2gen = 20 # number of new notes to generate
        new_melody_pitch = generate(modelPitch_loaded, melody4gen_pitch, pitch_to_ix, next_notes=notes2gen)
        new_melody_duration = generate(modelDuration_loaded, melody4gen_duration, duration_to_ix, next_notes=notes2gen)
        
        print('length of gen melody: ', len(new_melody_pitch))
        print('generated pitches: ', np.array(new_melody_pitch[40:]) )
    
        converted = convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
        
        song_name = standards[i][12:][:-4]
        print('-'*30)
        print('Generating over song: '+ song_name)
        print('-'*30)
        #converted.write('output/gen4eval/music'+str(j)+'.mid')
        converted.write('output/gen4eval/'+ song_name + '_gen.mid')
        
        j+=1
    
    
    #%% MGEval (paper Explicitly conditioned melody generation)
    
    
    # Evauluate generation with MGEval
    # look inside the function to add/remove metrics
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
    
    #generated_path = 'output/gen4eval/*.mid'
    #MGEresults = MGEval(training_path, generated_path, num_of_generations)


    #%% Melody Segmentation
    
    def separateSeqs(seq_pitch, seq_duration, segment_length = 35):
        # Separate the songs into single melodies in order to avoid 
        # full batches of pad tokens
        
        tot_pitch = []
        tot_duration = []
        new_pitch = []
        new_duration = []
        long_dur = ['full', 'half', 'quarter', 'dot half', 'dot quarter', 
                    'dot 8th', 'half note triplet', 'quarter note triplet']
        
        long_dur = ['full', 'half', 'quarter', 'dot half', 'dot quarter']
        
        counter = 0
        for i in range(min(len(seq_pitch), len(seq_duration))):
            new_pitch.append(seq_pitch[i])
            new_duration.append(seq_duration[i])
            counter += 1
            if seq_pitch[i] == 'R' and seq_duration[i] in long_dur:
                tot_pitch.append(np.array(new_pitch, dtype=object))
                tot_duration.append(np.array(new_duration, dtype=object))
                new_pitch = []
                new_duration = []
                counter = 0
            elif counter == segment_length:
                tot_pitch.append(np.array(new_pitch, dtype=object))
                tot_duration.append(np.array(new_duration, dtype=object))
                new_pitch = []
                new_duration = []
                counter = 0
        return tot_pitch, tot_duration
    
    def segmentDataset(pitch_data, duration_data, segment_length = 35):
        pitch_segmented = []
        duration_segmented = []
        for i in range(min(len(pitch_data), len(duration_data))):
            train_pitch_sep, train_duration_sep = separateSeqs(pitch_data[i], duration_data[i], segment_length)
            for seq in train_pitch_sep:
                pitch_segmented.append(seq)
            for seq in train_duration_sep:
                duration_segmented.append(seq)
        pitch_segmented = np.array(pitch_segmented, dtype=object)
        duration_segmented = np.array(duration_segmented, dtype=object)
        
        return pitch_segmented, duration_segmented
    
    # Maximum value of a sequence
    segment_length = 35
    train_pitch_segmented, train_duration_segmented = segmentDataset(train_pitch, train_duration, segment_length)
    val_pitch_segmented, val_duration_segmented = segmentDataset(val_pitch, val_duration, segment_length)
    test_pitch_segmented, test_duration_segmented = segmentDataset(test_pitch, test_duration, segment_length)


    #%% Perplexity, Test Loss, Accuracy
    
    #DATA PREPARATION FOR TEST 

    # pad data to max_lenght of sequences, prepend <sos> and append <eos>
    def pad(data):
        # from: https://pytorch.org/text/_modules/torchtext/data/field.html
        data = list(data)
        # calculate max lenght
        max_len = max(len(x) for x in data)
        # Define padding tokens
        pad_token = '<pad>'
        init_token = '<sos>'
        eos_token = '<eos>'
        # pad each sequence in the data to max_lenght
        padded, lengths = [], []
        for x in data:
            padded.append(
                #([init_token])
                #+ 
                list(x[:max_len])
                #+ ([eos_token])
                + [pad_token] * max(0, max_len - len(x)))
        lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        return padded
    
    # divide into batches of size bsz and converts notes into numbers
    def batchify(data, bsz, dict_to_ix):
        
        padded = pad(data)
        padded_num = [[dict_to_ix[x] for x in ex] for ex in padded]
        
        data = torch.tensor(padded_num, dtype=torch.long)
        data = data.contiguous()
        
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)
    
    batch_size = 20
    eval_batch_size = 10
    
    train_data_pitch = batchify(train_pitch_segmented, batch_size, pitch_to_ix)
    val_data_pitch = batchify(val_pitch_segmented, eval_batch_size, pitch_to_ix)
    test_data_pitch = batchify(test_pitch_segmented, eval_batch_size, pitch_to_ix)
    
    train_data_duration = batchify(train_duration_segmented, batch_size, duration_to_ix)
    val_data_duration = batchify(val_duration_segmented, eval_batch_size, duration_to_ix)
    test_data_duration = batchify(test_duration_segmented, eval_batch_size, duration_to_ix)
    
    # divide into target and input sequence of lenght bptt
    # --> obtain matrices of size bptt x batch_size
    bptt = segment_length # lenght of a sequence of data (IMPROVEMENT HERE!!)
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len] # input 
        target = source[i+1:i+1+seq_len].reshape(-1) # target (same as input but shifted by 1)
        targets_no_reshape = source[i+1:i+1+seq_len]
        return data, target, targets_no_reshape
    
    
    # Calculate loss and perplexity on a test set
    def lossPerplexityAccuracy(eval_model, data_source, vocab, criterion):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(vocab)
        # accuracy is calculated token-by-token
        tot_tokens = data_source.shape[0]*data_source.shape[1]
        correct = 0
        src_mask = eval_model.generate_square_subsequent_mask(bptt).to(device)
        
        tot_tokens = 0 # REMOVE
        
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                # get batch
                data, targets, targets_no_reshape = get_batch(data_source, i)
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
                        if targets_no_reshape[j,k] != 49 and targets_no_reshape[j,k] != 12: # REMOVE
                            correct += (max_idx == targets_no_reshape[j,k]).sum().item()
                            tot_tokens += 1
        
        accuracy = correct / tot_tokens *100
        loss = total_loss / (len(data_source) - 1)
        perplexity = math.exp(loss)
        return loss, perplexity, accuracy

    
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
    
    #MGEresults = MGEval(training_path, generated_path, num_of_generations)
    #metrics_result['MGEval'] = MGEresults
    
    criterion = nn.CrossEntropyLoss()
    perplexity_results_pitch, testLoss_results_pitch, accuracy_results_pitch  = lossPerplexityAccuracy(modelPitch_loaded, test_data_pitch, vocabPitch, criterion)
    metrics_result['Pitch_perplexity'] = perplexity_results_pitch
    metrics_result['Pitch_test-loss'] = testLoss_results_pitch
    metrics_result['Pitch_accuracy'] = accuracy_results_pitch
    
    perplexity_results_duration, testLoss_results_duration, accuracy_results_duration  = lossPerplexityAccuracy(modelDuration_loaded, test_data_duration, vocabDuration, criterion)
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
    
    