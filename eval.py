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
from dataset_funct import ImprovDurationDataset, ImprovPitchDataset, readMIDI, convertMIDI

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

# TRANSFORMER MODEL
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
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

def generate(model, melody4gen, dict_to_ix, next_notes=10):
    melody4gen = melody4gen.tolist()
    for i in range(0,next_notes):
        x_pred = torch.tensor([dict_to_ix[w] for w in melody4gen], dtype=torch.long)
        y_pred = model(x_pred)
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        melody4gen.append(getNote(word_index, dict_to_ix))
    return melody4gen

if __name__ == '__main__':
    
    # DATA LOADING
    
    # LOAD PITCH DATASET
    datasetPitch = ImprovPitchDataset()
    X_pitch = datasetPitch.getData()
    # set vocabulary for conversion
    vocabPitch = datasetPitch.vocab
    # Add padding tokens to vocab
    vocabPitch.append('<pad>')
    vocabPitch.append('<sos>')
    vocabPitch.append('<eos>')
    pitch_to_ix = {word: i for i, word in enumerate(vocabPitch)}
    
    # Divide pitch into train, validation and test
    train_pitch = X_pitch[:int(len(X_pitch)*0.7)]
    val_pitch = X_pitch[int(len(X_pitch)*0.7)+1:int(len(X_pitch)*0.7)+1+int(len(X_pitch)*0.1)]
    test_pitch = X_pitch[int(len(X_pitch)*0.7)+1+int(len(X_pitch)*0.1):]
    
    # LOAD DURATION DATASET
    datasetDuration = ImprovDurationDataset()
    X_duration = datasetDuration.getData()
    # set vocabulary for conversion
    vocabDuration = datasetDuration.vocab
    # Add padding tokens to vocab
    vocabDuration.append('<pad>')
    vocabDuration.append('<sos>')
    vocabDuration.append('<eos>')
    duration_to_ix = {word: i for i, word in enumerate(vocabDuration)}
    
    # Divide duration into train, validation and test
    train_duration = X_duration[:int(len(X_duration)*0.7)]
    val_duration = X_duration[int(len(X_duration)*0.7)+1:int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1)]
    test_duration = X_duration[int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1):]
    
    #%% IMPORT PRE-TRAINED MODEL
    
    # HYPERPARAMETERS
    ntokens_duration = len(vocabDuration) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    modelDuration_loaded = TransformerModel(ntokens_duration, emsize, nhead, nhid, nlayers, dropout).to(device)

    # Import model
    savePATHduration = 'modelsDuration/modelDuration_10epochs_padding.pt'
    modelDuration_loaded.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))
    
    # HYPERPARAMETERS
    ntokens_pitch = len(vocabPitch) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    modelPitch_loaded = TransformerModel(ntokens_pitch, emsize, nhead, nhid, nlayers, dropout).to(device)

    # Import model
    savePATHpitch = 'modelsPitch/modelPitch_10epochs_padding.pt'
    modelPitch_loaded.load_state_dict(torch.load(savePATHpitch, map_location=torch.device('cpu')))
    
    
    #%% SAMPLES GENERATION WITH PRE-TRAINED MODELS

    # Remove characters who are not in the dictionary
    def onlyDict(pitchs, durations, vocabPitch, vocabDuration):
        # takes an array and a dictionary and gives the same array without
        # the elements who are not in the dictionary
        new_pitch = []
        new_duration = []
        for i in range(len(pitchs)):
            if pitchs[i] in vocabPitch and durations[i] in vocabDuration:
                new_pitch.append(pitchs[i]) 
                new_duration.append(durations[i]) 
        new_pitch = np.array(new_pitch) # same solos but with only most frequent notes
        new_duration = np.array(new_duration) # same solos but with only most frequent notes
        return new_pitch, new_duration
    
    
    #specify the path
    f = 'data/w_jazz/JohnColtrane_Mr.P.C._FINAL.mid'
    melody4gen_pitch, melody4gen_duration, dur_dict, song_properties = readMIDI(f)
    melody4gen_pitch, melody4gen_duration = onlyDict(melody4gen_pitch, melody4gen_duration, vocabPitch, vocabDuration)
    melody4gen_pitch = melody4gen_pitch[:80]
    melody4gen_duration = melody4gen_duration[:80]
    #print(melody4gen_pitch)
    #print(melody4gen_duration)
    
    notes2gen = 40 # number of new notes to generate
    new_melody_pitch = generate(modelPitch_loaded, melody4gen_pitch, pitch_to_ix, next_notes=notes2gen)
    new_melody_duration = generate(modelDuration_loaded, melody4gen_duration, duration_to_ix, notes2gen)

    converted = convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
    converted.write('output/music.mid')
    
    
    #%% MGEval (paper Explicitly conditioned melody generation)
    
    # Evauluate generation with MGEval
    # look inside the function to add/remove metrics
    def MGEval(training_midi_path, generated_midi_path, num_samples = 20):
        
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
        
        # Initialize dataset1 (training data)
        set1 = glob.glob(training_midi_path)
        # Dictionary of metrics
        set1_eval = {'total_used_pitch':np.zeros((num_samples,1))}
        # Add metrics to the dictionary
        set1_eval['total_pitch_class_histogram'] = np.zeros((num_samples,12))
        #set1_eval['total_used_note'] = np.zeros((num_samples,1))
        set1_eval['pitch_class_transition_matrix'] = np.zeros((num_samples,12,12))
        set1_eval['pitch_range'] = np.zeros((num_samples,1))
        #set1_eval['avg_pitch_shift'] = np.zeros((num_samples,1))
        set1_eval['avg_IOI'] = np.zeros((num_samples,1))
        #set1_eval['note_length_hist'] = np.zeros((num_samples,12))
        
        # Calculate metrics
        metrics_list = list(set1_eval.keys())
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
            print(metrics_list[i] + ':')
            print('------------------------')
            print(' demo_set1')
            print('  Kullback–Leibler divergence:',utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i]))
            print('  Overlap area:', utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i]))
            
            print(' demo_set2')
            print('  Kullback–Leibler divergence:',utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i]))
            print('  Overlap area:', utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i]))
            print()
    
    
    training_path = 'data/w_jazz/*.mid'
    
    # BUILD A DATASET OF GENERATED SEQUENCES
    import glob
    standards = glob.glob(training_path)
    num_of_generations = 20
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
        new_melody_duration = generate(modelDuration_loaded, melody4gen_duration, duration_to_ix, notes2gen)
    
        converted = convertMIDI(new_melody_pitch, new_melody_duration, song_properties['tempo'], dur_dict)
        
        song_name = standards[i][12:][:-4]
        print('-'*30)
        print('Generating over song: '+ song_name)
        print('-'*30)
        #converted.write('output/gen4eval/music'+str(j)+'.mid')
        converted.write('output/gen4eval/'+ song_name + '_gen.mid')
        
        j+=1
        
    
    generated_path = 'output/gen4eval/*.mid'
    
    MGEval(training_path, generated_path, num_of_generations)
    
    #%% MODEL EVALUATION
    # accuracy, perplexity (paper seq-Attn)
    # NLL loss, BLEU (paper explicitly conditioned melody generation)
    
    #parameters = []
    #for param in modelDuration_loaded.parameters():
        #parameters.append(param.data.numpy())
        #print(param.data)
    
    """
    BLEU code: il always gets 0, don't understand why
    
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
    
    