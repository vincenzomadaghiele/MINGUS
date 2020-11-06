#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:47:05 2020

@author: vincenzomadaghiele

This scripts recreates the 'no_cond' model of the paper:
    Explicitly Conditioned Melody Generation: A Case Study with Interdependent RNNs
    (Genchel et al., 2019)
but using Transformers instead of RNNs
and with a different dataset
"""

# library for understanding music
from music21 import *
import torch
import torch.nn as nn
import numpy as np
import math
from dataset_funct import ImprovDurationDataset, ImprovPitchDataset, convert_to_midi, read_midi_pitch, read_midi_duration 

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

import time
def train(model, vocab, train_data, criterion, optimizer):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(vocab)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source, vocab):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(vocab)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


if __name__ == '__main__':

    #%% DATA LOADING
    
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
    #print(X_pitch[:3])
    
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
    #print(X_duration[:3])
    
    # Divide duration into train, validation and test
    train_duration = X_duration[:int(len(X_duration)*0.7)]
    val_duration = X_duration[int(len(X_duration)*0.7)+1:int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1)]
    test_duration = X_duration[int(len(X_duration)*0.7)+1+int(len(X_duration)*0.1):]
    
    
    
    #%% DATA PREPARATION

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
                ([init_token])
                + list(x[:max_len])
                + ([eos_token])
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
    
    train_data_pitch = batchify(train_pitch, batch_size, pitch_to_ix)
    val_data_pitch = batchify(val_pitch, eval_batch_size, pitch_to_ix)
    test_data_pitch = batchify(test_pitch, eval_batch_size, pitch_to_ix)
    
    train_data_duration = batchify(train_duration, batch_size, duration_to_ix)
    val_data_duration = batchify(val_duration, eval_batch_size, duration_to_ix)
    test_data_duration = batchify(test_duration, eval_batch_size, duration_to_ix)
    
    # divide into target and input sequence of lenght bptt
    # --> obtain matrices of size bptt x batch_size
    bptt = 35 # lenght of a sequence of data (IMPROVEMENT HERE!!)
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len] # input 
        target = source[i+1:i+1+seq_len].view(-1) # target (same as input but shifted by 1)
        return data, target
    
    #%% PITCH MODEL TRAINING
    
    # HYPERPARAMETERS
    ntokens_pitch = len(vocabPitch) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    modelPitch = TransformerModel(ntokens_pitch, emsize, nhead, nhid, nlayers, dropout).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 100 # The number of epochs
    best_model = None
    
    # TRAINING LOOP
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(modelPitch, vocabPitch, train_data_pitch, criterion, optimizer)
        val_loss = evaluate(modelPitch, val_data_pitch, vocabPitch)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_pitch = modelPitch
    
        scheduler.step()
    
    # TEST THE MODEL
    test_loss = evaluate(best_model_pitch, test_data_pitch, vocabPitch)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    
    savePATHpitch = 'modelsPitch/modelPitch_'+ str(epochs) + 'epochs_padding.pt'
    state_dictPitch = modelPitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)
    

    #%% DURATION MODEL TRAINING
    
    # HYPERPARAMETERS
    ntokens_duration = len(vocabDuration) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    modelDuration = TransformerModel(ntokens_duration, emsize, nhead, nhid, nlayers, dropout).to(device)
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(modelDuration.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # TRAIN AND EVALUATE LOSS
    best_val_loss = float("inf")
    epochs = 100 # The number of epochs
    best_model = None
    
    # TRAINING LOOP
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(modelDuration, vocabDuration, train_data_duration, criterion, optimizer)
        val_loss = evaluate(modelDuration, val_data_duration, vocabDuration)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_duration = modelDuration
    
        scheduler.step()
    
    # TEST THE MODEL
    test_loss = evaluate(best_model_duration, test_data_duration, vocabDuration)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    
    savePATHduration = 'modelsDuration/modelDuration_'+ str(epochs) + 'epochs_padding.pt'
    state_dictDuration = modelDuration.state_dict()
    torch.save(state_dictDuration, savePATHduration)
    

    #%% SAMPLES GENERATION

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
    
    #specify the path
    f='data/w_jazz/CharlieParker_DonnaLee_FINAL.mid'
    melody4gen_pitch = (read_midi_pitch(f))[:80]
    melody4gen_duration = (read_midi_duration(f))[:80]
    
    notes2gen = 40 # number of new notes to generate
    new_melody_pitch = generate(modelPitch, melody4gen_pitch, pitch_to_ix, next_notes=notes2gen)
    new_melody_duration = generate(modelDuration, melody4gen_duration, duration_to_ix, notes2gen)

    midi_gen = convert_to_midi(new_melody_pitch, new_melody_duration)
    midi_gen.write('midi', fp='output/gen4eval/music.mid') 
    
    
    # SAVE MODELS
    
    savePATHpitch = 'modelsPitch/modelPitch_'+ str(epochs) + 'epochs_padding.pt'
    state_dictPitch = modelPitch.state_dict()
    torch.save(state_dictPitch, savePATHpitch)

    savePATHduration = 'modelsDuration/modelDuration_'+ str(epochs) + 'epochs_padding.pt'
    state_dictDuration = modelDuration.state_dict()
    torch.save(state_dictDuration, savePATHduration)
    
    
    #%% MODEL EVALUATION
    # accuracy, perplexity (paper seq-Attn)
    # NLL loss, BLEU, MGeval (paper explicitly conditioned melody generation)
    
    def bleu_score():
        pass
    
    def perplexity():
        pass
    
    def accuracy_score(y_true, y_pred):
        y_pred = np.concatenate(tuple(y_pred))
        y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(y_pred.shape)
        return (y_true == y_pred).sum() / float(len(y_true))
    
    def evaluateAll(test_data, model, dict_to_ix):
        eval_metrics = []
        y_true = []
        y_pred = []
        for seq in test_data:
            # Not doable: change approach !!!
            for i in range(1000, len(seq)-1):
                seq4pred = np.array(seq[:i])
                target = seq[i+1]
                target_hat = generate(model, seq4pred, dict_to_ix)[-1]
                y_true.append(target)
                y_pred.append(target_hat)
        accuracy = accuracy_score(y_true, y_pred)
        eval_metrics['accuracy'] = accuracy
        return eval_metrics

    #evaluateModelPitch = evaluateAll(test_pitch[0], modelPitch, pitch_to_ix)
    
    #%% MGEval 
    
    
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
        set1_eval['total_used_note'] = np.zeros((num_samples,1))
        set1_eval['pitch_class_transition_matrix'] = np.zeros((num_samples,12,12))
        set1_eval['pitch_range'] = np.zeros((num_samples,1))
        #set1_eval['avg_pitch_shift'] = np.zeros((num_samples,1))
        set1_eval['avg_IOI'] = np.zeros((num_samples,1))
        set1_eval['note_length_hist'] = np.zeros((num_samples,12))
        
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
        set2_eval['total_used_note'] = np.zeros((num_samples,1))
        set2_eval['pitch_class_transition_matrix'] = np.zeros((num_samples,12,12))
        set2_eval['pitch_range'] = np.zeros((num_samples,1))
        #set2_eval['avg_pitch_shift'] = np.zeros((num_samples,1))
        set2_eval['avg_IOI'] = np.zeros((num_samples,1))
        set2_eval['note_length_hist'] = np.zeros((num_samples,12))
        
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
     

    # Check if it works next time and if it does delete this
    # Convert the output of the model to midi
    def convert_to_midi(new_melody_pitch, new_melody_duration):
        offset = 0
        output_notes = []
        #output_notes = stream.Stream()
        # create note and chord objects based on the values generated by the model
        for i in range(len(new_melody_pitch)):
            if new_melody_pitch[i] == 'R' and new_melody_pitch[i] != '<sos>' and new_melody_pitch[i] != '<eos>' and new_melody_duration[i] != '<sos>' and new_melody_duration[i] != '<eos>':
                new_note = note.Rest()
                new_note.offset = offset
                #new_note.duration.quarterLength = float(new_melody_duration[i])
            elif new_melody_pitch[i] != '<sos>' and new_melody_pitch[i] != '<eos>' and new_melody_duration[i] != '<sos>' and new_melody_duration[i] != '<eos>': 
                new_note = note.Note()
                new_note.pitch.nameWithOctave = new_melody_pitch[i]
                new_note.offset = offset
                #new_note.duration.quarterLength = float(new_melody_duration[i])
                new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            if new_melody_duration[i] != '<sos>' and new_melody_duration[i] != '<eos>': 
                offset += float(new_melody_duration[i])
            
        midi_stream = stream.Stream(output_notes)
        return midi_stream
        
    # Remove characters who are not in the dictionary
    def onlyDict(array, vocab):
        # takes an array and a dictionary and gives the same array without
        # the elements who are not in the dictionary
        new_array=[]
        for note in array:
            if note in vocab:
                new_array.append(note)            
        new_array = np.array(new_array) # same solos but with only most frequent notes
        return new_array
    
    
    training_path = 'data/w_jazz_mini/*.mid'
    
    # Build a dataset of generated samples
    import glob
    standards = glob.glob(training_path)
    num_of_generations = 10
    j=0
    for i in range(0, num_of_generations):
        
        print(standards[i])
        melody4gen_pitch = (read_midi_pitch(standards[i]))[:40]
        melody4gen_pitch = onlyDict(melody4gen_pitch, vocabPitch)
        melody4gen_duration = (read_midi_duration(standards[i]))[:40]
        melody4gen_duration = onlyDict(melody4gen_duration, vocabDuration)
        
        # in case one sequence has less elements than another one
        if len(melody4gen_pitch) > len(melody4gen_duration):
            diff = len(melody4gen_pitch) - len(melody4gen_duration)
            melody4gen_pitch = melody4gen_pitch[:diff]
        elif len(melody4gen_pitch) < len(melody4gen_duration):
            diff = len(melody4gen_duration) - len(melody4gen_pitch)
            melody4gen_pitch = melody4gen_duration[:diff]
        
        notes2gen = 20 # number of new notes to generate
        new_melody_pitch = generate(modelPitch, melody4gen_pitch, pitch_to_ix, next_notes=notes2gen)
        new_melody_duration = generate(modelDuration, melody4gen_duration, duration_to_ix, notes2gen)
    
        midi_gen = convert_to_midi(new_melody_pitch, new_melody_duration)
        midi_gen.write('midi', fp='output/gen4eval/music'+str(j)+'.mid') 
        j+=1
        
    generated_path = 'output/gen4eval/*.mid'

    MGEval(training_path, generated_path, num_of_generations)
    
    
    
    