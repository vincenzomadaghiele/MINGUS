#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:37:30 2021

@author: vincenzomadaghiele
"""

import numpy as np
import torch
import torch.nn as nn
import math
import time
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter


# TRANSFORMER MODEL
class TransformerModel(nn.Module):

    def __init__(self, pitch_vocab_size, pitch_embed_dim,
                     duration_vocab_size, duration_embed_dim, 
                     bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
                     beat_vocab_size, beat_embed_dim, 
                     offset_vocab_size, offset_embed_dim, 
                     ninp, nhead, nhid, nlayers, 
                     pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
                     device, dropout=0.5, isPitch=True, COND='I-C-NC-B-BE-O'):
        
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pitch_pad_idx = pitch_pad_idx
        self.duration_pad_idx = duration_pad_idx
        self.beat_pad_idx = beat_pad_idx
        self.offset_pad_idx = offset_pad_idx
        self.device = device
        self.ninp = ninp
        self.isPitch = isPitch
        
        #COND = COND.split("-")
        self.COND = COND.split("-")
        
        # feature embeddings
        self.pitch_embedding = nn.Embedding(pitch_vocab_size, pitch_embed_dim, padding_idx=self.pitch_pad_idx) # pitch
        self.duration_embedding = nn.Embedding(duration_vocab_size, duration_embed_dim, padding_idx=self.duration_pad_idx) # duration
        self.bass_embedding = nn.Embedding(pitch_vocab_size, bass_embed_dim, padding_idx=self.pitch_pad_idx) # bass
        self.beat_embedding = nn.Embedding(beat_vocab_size, beat_embed_dim, padding_idx=self.beat_pad_idx) # beat
        self.offset_embedding = nn.Embedding(offset_vocab_size, offset_embed_dim, padding_idx=self.offset_pad_idx) # beat
        
        
        self.chord_encoder = nn.Linear(4 * pitch_embed_dim, chord_encod_dim)
        
        #encoder_input_dim = 2 * pitch_embed_dim + duration_embed_dim + chord_encod_dim #+ beat_embed_dim
        self.next_chord_encoder = nn.Linear(4 * pitch_embed_dim, next_chord_encod_dim)
        
        # Try out chord embeds
        # chord embeds require a new chord dictionary or a linear layer!
        #chord_embed_dim = 64
        #self.chord_emedding = nn.Embedding(pitch_vocab_size, chord_embed_dim, padding_idx=self.pitch_pad_idx) 
        
        if self.isPitch:
            encoder_input_dim = pitch_embed_dim
        else:
            encoder_input_dim = duration_embed_dim
        if 'I' in self.COND:
            if self.isPitch:
                encoder_input_dim += duration_embed_dim
            else:
                encoder_input_dim += pitch_embed_dim
        if 'C' in self.COND:
            encoder_input_dim += chord_encod_dim
        if 'NC' in self.COND:
            encoder_input_dim += next_chord_encod_dim
        if 'B' in self.COND:
            encoder_input_dim += pitch_embed_dim
        if 'BE' in self.COND:
            encoder_input_dim += beat_embed_dim
        if 'O' in self.COND:
            encoder_input_dim += offset_embed_dim

        
        #encoder_input_dim = 2 * pitch_embed_dim + duration_embed_dim + chord_encod_dim + next_chord_encod_dim + beat_embed_dim + offset_embed_dim 
        #print(encoder_input_dim)
        
        # Start the transformer structure with multidimensional data
        #encoder_input_dim = 6 * pitch_embed_dim + duration_embed_dim #+ beat_embed_dim
        self.encoder = nn.Linear(encoder_input_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        if self.isPitch:
            self.out_linear = nn.Linear(ninp, pitch_embed_dim)
            self.out_decoder = nn.Linear(pitch_embed_dim, pitch_vocab_size)
        else:
            self.out_linear = nn.Linear(ninp, duration_embed_dim)
            self.out_decoder = nn.Linear(duration_embed_dim, duration_vocab_size)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_src_pad_mask(self, src):
        pad_mask = src.transpose(0, 1) == self.src_pad_idx
        #pad_mask = pad_mask.float().masked_fill(pad_mask == True, float('-inf')).masked_fill(pad_mask == False, float(0.0))
        return pad_mask.to(self.device)

    def init_weights(self):
        initrange = 0.1
        
        # initialize embedding weights
        self.pitch_embedding.weight.data.uniform_(-initrange, initrange)
        self.duration_embedding.weight.data.uniform_(-initrange, initrange)
        self.bass_embedding.weight.data.uniform_(-initrange, initrange)
        self.beat_embedding.weight.data.uniform_(-initrange, initrange)
        self.offset_embedding.weight.data.uniform_(-initrange, initrange)
        
        # initialize linear layers weigths
        self.chord_encoder.bias.data.zero_()
        self.chord_encoder.weight.data.uniform_(-initrange, initrange)
        self.next_chord_encoder.bias.data.zero_()
        self.next_chord_encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.out_linear.bias.data.zero_()
        self.out_linear.weight.data.uniform_(-initrange, initrange)
        self.out_decoder.bias.data.zero_()
        self.out_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, pitch, duration, chord, next_chord, bass, beat, offset, src_mask):
        
        # embed data
        pitch_embeds = self.pitch_embedding(pitch)
        duration_embeds = self.duration_embedding(duration)
        chord_embeds = self.pitch_embedding(chord).view(chord.shape[0], chord.shape[1], -1).contiguous()
        next_chord_embeds = self.pitch_embedding(next_chord).view(next_chord.shape[0], next_chord.shape[1], -1).contiguous()
        bass_embeds = self.pitch_embedding(bass)
        beat_embeds = self.beat_embedding(beat)
        offset_embeds = self.offset_embedding(offset)

        #chord_embeds = self.chord_emedding(chord).view(chord.shape[0], chord.shape[1], -1).contiguous()
        chord_embeds = self.chord_encoder(chord_embeds)
        next_chord_embeds = self.next_chord_encoder(next_chord_embeds)

        if self.isPitch:
            src = pitch_embeds
        else:
            src = duration_embeds
        if 'I' in self.COND:
            if self.isPitch:
                src = (torch.cat([src, duration_embeds], 2))
            else:
                src = (torch.cat([src, pitch_embeds], 2))
        if 'C' in self.COND:
            src = (torch.cat([src, chord_embeds], 2))
        if 'NC' in self.COND:
            src = (torch.cat([src, next_chord_embeds], 2))
        if 'B' in self.COND:
            src = (torch.cat([src, bass_embeds], 2))
        if 'BE' in self.COND:
            src = (torch.cat([src, beat_embeds], 2))
        if 'O' in self.COND:
            src = (torch.cat([src, offset_embeds], 2))
            
        src = self.encoder(src) * math.sqrt(self.ninp)


        # Concatenate along 3rd dimension
        #src = self.encoder(torch.cat([pitch_embeds, duration_embeds, bass_embeds, chord_embeds], 2)) * math.sqrt(self.ninp)
        #src = self.encoder(torch.cat([pitch_embeds, duration_embeds, 
        #                              bass_embeds, chord_embeds, 
        #                              next_chord_embeds, beat_embeds, 
        #                              offset_embeds], 2)) * math.sqrt(self.ninp)
        
        
        #src_padding_mask = self.make_src_pad_mask(src) # PROBLEM
        
        # Positional encoding
        src = self.pos_encoder(src)
        #output = self.transformer_encoder(src, src_mask, src_padding_mask)
        output = self.transformer_encoder(src, src_mask)
        output = self.out_linear(output)
        output = self.out_decoder(output)
        
        return output
    
    def getEmbeds(self, pitch, duration, chord, bass, beat):
        # embedded data
        pitch_embeds = self.pitch_embedding(pitch)
        duration_embeds = self.duration_embedding(duration)
        return pitch_embeds, duration_embeds


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


def train(model, vocabTarget, 
          train_data_pitch, train_data_duration,  
          train_data_chord, train_data_next_chord,
          train_data_bass, train_data_beat, train_data_offset, 
          criterion, optimizer, scheduler, epoch, bptt, device, 
          writer, step, isPitch=True):
    
    '''

    Parameters
    ----------
    model : pytorch Model
        Model to be trained.
    vocabVelocity : python dictionary
        dictionary of the tokens used in training.
    train_data_velocity : pytorch Tensor
        batched velocity data to be used in training.
        Same size of pitch and duration.
    train_data_pitch : pytorch Tensor
        batched pitch data to be used in training.
        Same size of velocity and duration.
    train_data_duration : pytorch Tensor
        batched duration data to be used in training.
        Same size of velocity and pitch.
    criterion : pytorch Criterion
        criterion to be used for otimization.
    optimizer : pytorch Optimizer
        optimizer used in training.

    Returns
    -------
    None.

    '''
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(vocabTarget)
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data_pitch.size(0) - 1, bptt)):
        
        if isPitch:
            # get a batch of data
            data_pitch, targets, _ = get_batch(train_data_pitch, i, bptt)
            data_duration, _, _ = get_batch(train_data_duration, i, bptt)
        else:
            data_pitch, _, _ = get_batch(train_data_pitch, i, bptt)
            data_duration, targets, _ = get_batch(train_data_duration, i, bptt)
        data_chord, _, _ = get_batch(train_data_chord, i, bptt)
        data_next_chord, _, _ = get_batch(train_data_next_chord, i, bptt)
        data_bass, _, _ = get_batch(train_data_bass, i, bptt)
        data_beat, _, _ = get_batch(train_data_beat, i, bptt)
        data_offset, _, _ = get_batch(train_data_offset, i, bptt)
        
        
        if data_pitch.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data_pitch.size(0)).to(device)
        
        # pass the data trought the model
        output = model(data_pitch, data_duration, data_chord, data_next_chord,
                       data_bass, data_beat, data_offset, src_mask) 
        loss = criterion(output.view(-1, ntokens), targets)
        
        # backpropagation step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # update losses
        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data_pitch) // bptt, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        
        if i == 0:
            # Visualize embeddings
            if isPitch:
                pitch_embeds, _ = model.getEmbeds(data_pitch, data_duration, data_chord,
                                                  data_bass, data_beat)
                features = pitch_embeds.reshape(-1, pitch_embeds.shape[2])
                pitch_classes = data_pitch.cpu().reshape(-1).numpy()
                class_labels = [vocabTarget[pitch] for pitch in pitch_classes]
                #writer.add_embedding(features, metadata=class_labels, global_step=step)
            else:
                _, dur_embeds = model.getEmbeds(data_pitch, data_duration, data_chord,
                                                  data_bass, data_beat)
                features = dur_embeds.reshape(-1, dur_embeds.shape[2])
                dur_classes = data_duration.cpu().reshape(-1).numpy()
                class_labels = [vocabTarget[duration] for duration in dur_classes]
                #writer.add_embedding(features, metadata=class_labels, global_step=epoch)
            if epoch == 1:
                writer.add_graph(model, input_to_model = (data_pitch, data_duration, data_chord, data_next_chord,
                                                          data_bass, data_beat, data_offset, src_mask))
            
        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1

    return step


def evaluate(eval_model, vocabTarget, 
             eval_data_pitch, eval_data_duration, 
             eval_data_chord, eval_data_next_chord, 
             eval_data_bass, eval_data_beat, eval_data_offset,
             criterion, bptt, device, isPitch=True):
    '''

    Parameters
    ----------
    vocabVelocity : python dictionary
        dictionary of the tokens used in training.
    eval_data_velocity : pytorch Tensor
        batched velocity data to be used in training.
        Same size of pitch and duration.
    eval_data_pitch : pytorch Tensor
        batched pitch data to be used in training.
        Same size of velocity and duration.
    eval_data_duration : pytorch Tensor
        batched duration data to be used in training.
        Same size of velocity and pitch.
    eval_model : pytorch Model
        Model, already trained to be evaluated.

    Returns
    -------
    float
        total loss of the model on validation/test data.

    '''
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    total_accuracy = 0.
    tot_tokens = 0.
    ntokens = len(vocabTarget)
    src_mask = eval_model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data_pitch.size(0) - 1, bptt):
            
            # get a batch of data
            if isPitch:
                data_pitch, targets, _ = get_batch(eval_data_pitch, i, bptt)
                data_duration, _, _ = get_batch(eval_data_duration, i, bptt)
            else:
                data_pitch, _, _ = get_batch(eval_data_pitch, i, bptt)
                data_duration, targets, _ = get_batch(eval_data_duration, i, bptt)
            data_chord, _, _ = get_batch(eval_data_chord, i, bptt)
            data_next_chord, _, _ = get_batch(eval_data_next_chord, i, bptt)
            data_bass, _, _ = get_batch(eval_data_bass, i, bptt)
            data_beat, _, _ = get_batch(eval_data_beat, i, bptt)
            data_offset, _, _ = get_batch(eval_data_offset, i, bptt)
            
            if data_pitch.size(0) != bptt:
                src_mask = eval_model.generate_square_subsequent_mask(data_pitch.size(0)).to(device)
            
            output = eval_model(data_pitch, data_duration, data_chord, data_next_chord,
                                data_bass, data_beat, data_offset, src_mask) 
            
            # cross-entropy loss
            output_flat = output.view(-1, ntokens)
            total_loss += len(data_pitch) * criterion(output_flat, targets).item()
            
            # accuracy
            padTokent = vocabTarget['<pad>']
            max_logprobs = np.argmax(output_flat.cpu().numpy(), axis=1)
            nptargets = np.copy(targets.cpu().numpy())
            # ensure that pad tokens are not counted
            nptargets[nptargets == padTokent] = 1000
            # count not padding elements
            notPaddingElements = (nptargets != 1000).sum()
            
            total_accuracy += accuracy_score(max_logprobs, nptargets) * notPaddingElements
            tot_tokens += notPaddingElements
            
    return total_loss / (len(eval_data_pitch) - 1), total_accuracy / tot_tokens


def get_batch(source, i, bptt):
    '''

    Parameters
    ----------
    source : pytorch Tensor
        source of batched data.
    i : integer
        number of the batch to be selected.

    Returns
    -------
    data : pytorch Tensor
        batch of data of size [bptt x bsz].
    target : pytorch Tensor
        same as data but sequences are shifted by one token, 
        of size [bptt x bsz].

    '''
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len] # input 
    target = source[i+1:i+1+seq_len].view(-1) # target (same as input but shifted by 1)
    targets_no_reshape = source[i+1:i+1+seq_len]

    return data, target, targets_no_reshape
