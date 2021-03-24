#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 07:09:14 2021

@author: vincenzomadaghiele
"""

import pretty_midi
import numpy as np
import torch
import torch.nn as nn
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

# This is used in the generate() function
def batch4gen(data, bsz, dict_to_ix, device, isChord=False):
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
    
    if isChord:
        padded_num = [[dict_to_ix[note] for note in chord] for chord in data]
    else:
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

def generateCond(tune, num_bars, temperature, modelPitch, modelDuration):
        
    # select a tune
    #tune = structuredSongs[0]
    # choose how many starting bars
    #num_bars = 4
    # define temperature
    #temperature = 1   
    
    # copy bars into a new_structured_song
    new_structured_song = {}
    new_structured_song['title'] = tune['title'] + '_gen'
    new_structured_song['total time [sec]'] = tune['total time [sec]']
    new_structured_song['quantization [sec]'] = tune['quantization [sec]']
    new_structured_song['quantization [beat]'] = tune['quantization [beat]']
    new_structured_song['tempo'] = tune['tempo']
    new_structured_song['beat duration [sec]'] = tune['beat duration [sec]']
    beat_duration_sec = new_structured_song['beat duration [sec]'] 
    bars = []
    beats = []
    
    print('Generating over song %s' % (tune['title']))
    
    # define duration dictionary
    unit = beat_duration_sec * 4 / 96.
    # possible note durations in seconds 
    # (it is possible to add representations - include 32nds, quintuplets...):
    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
    possible_durations = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3,
                          unit * 72, unit * 36, unit * 18, unit * 9, 
                          unit * 32, unit * 16, unit * 8, unit * 4]

    # Define durations dictionary
    dur_dict = {}
    dur_dict[possible_durations[0]] = 'full'
    dur_dict[possible_durations[1]] = 'half'
    dur_dict[possible_durations[2]] = 'quarter'
    dur_dict[possible_durations[3]] = '8th'
    dur_dict[possible_durations[4]] = '16th'
    dur_dict[possible_durations[5]] = '32th'
    dur_dict[possible_durations[6]] = 'dot half'
    dur_dict[possible_durations[7]] = 'dot quarter'
    dur_dict[possible_durations[8]] = 'dot 8th'
    dur_dict[possible_durations[9]] = 'dot 16th'
    dur_dict[possible_durations[10]] = 'half note triplet'
    dur_dict[possible_durations[11]] = 'quarter note triplet'
    dur_dict[possible_durations[12]] = '8th note triplet'
    dur_dict[possible_durations[13]] = '16th note triplet'
    inv_dur_dict = {v: k for k, v in dur_dict.items()}
    
    # initialize counters
    bar_num = num_bars
    beat_counter = 0
    offset_sec = 0
    beat_num = 0
    for bar in tune['bars'][:num_bars]:
        bars.append(bar)
        for beat in bar['beats']:
            beat_counter += 1
            #beat_num = beat['num beat'] - 1
            for duration in beat['duration']:
                duration_sec = inv_dur_dict[duration]
                offset_sec += duration_sec
    
    beat_pitch = []
    beat_duration = []
    next_beat_sec = (beat_counter + 1) * beat_duration_sec 
    
    # extract starting bars pitch, duration, chord, bass and put into array
    pitch = []
    duration = []
    chord = []
    bass = []
    for bar in bars:
        for beat in bar['beats']:
            for i in range(len(beat['pitch'])):
                pitch.append(beat['pitch'][i])
                duration.append(beat['duration'][i])
                chord.append(NottinghamToMidiChords[beat['chord']][:4])
                bass.append(NottinghamToMidiChords[beat['chord']][0])
    
    while len(bars) < len(tune['bars']):
        
        # batchify
        pitch4gen = batch4gen(np.array(pitch), len(pitch), pitch_to_ix, device)
        duration4gen = batch4gen(np.array(duration), len(duration), duration_to_ix, device)
        chord4gen = batch4gen(chord, len(chord), pitch_to_ix, device, isChord=True)
        bass4gen = batch4gen(bass, len(bass), pitch_to_ix, device)
        # reshape to column vectors
        pitch4gen = pitch4gen.t()
        duration4gen = duration4gen.t()
        chord4gen = chord4gen.t()
        chord4gen = chord4gen.reshape(chord4gen.shape[0], 1,chord4gen.shape[1])
        bass4gen = bass4gen.t()
        
        # generate new note conditioning on old arrays
        modelPitch.eval()
        modelDuration.eval()
        src_mask_pitch = modelPitch.generate_square_subsequent_mask(con.BPTT).to(device)
        src_mask_duration = modelDuration.generate_square_subsequent_mask(con.BPTT).to(device)
        if pitch4gen.size(0) != con.BPTT:
            src_mask_pitch = modelPitch.generate_square_subsequent_mask(pitch4gen.size(0)).to(device)
            src_mask_duration = modelDuration.generate_square_subsequent_mask(duration4gen.size(0)).to(device)
        
        # generate new pitch note
        pitch_pred = modelPitch(pitch4gen, duration4gen, chord4gen,
                                bass4gen, None, src_mask_pitch)
        word_weights = pitch_pred[-1].squeeze().div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0].item()
        new_pitch = vocabPitch[word_idx]
        # generate new duration note
        duration_pred = modelDuration(pitch4gen, duration4gen, chord4gen,
                                   bass4gen, None, src_mask_duration)
        word_weights = duration_pred[-1].squeeze().div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0].item()
        new_duration = vocabDuration[word_idx]
        
        # append note to new arrays
        beat_pitch.append(new_pitch)
        beat_duration.append(new_duration)
        duration_sec = inv_dur_dict[new_duration]
        offset_sec += duration_sec
        
        # check if the note is in a new beat / bar
        while offset_sec >= next_beat_sec:
            if beat_num >= 3:
                # end of bar
                # append beat
                beat = {}
                beat['num beat'] = beat_num + 1
                # check for chords
                new_chord = tune['bars'][bar_num]['beats'][beat_num]['chord']
                beat['chord'] = new_chord
                beat['pitch'] = beat_pitch 
                beat['duration'] = beat_duration 
                beat['offset'] = []
                beat['scale'] = []
                beat['bass'] = []
                beats.append(beat)
                beat_pitch = []
                beat_duration = []
                # append bar
                bar = {}
                bar['num bar'] = bar_num + 1 # over all song
                bar['beats'] = beats # beats 1,2,3,4
                bars.append(bar)
                beats = []
                beat_num = 0
                bar_num += 1
                beat_counter += 1
                next_beat_sec = (beat_counter + 1) * beat_duration_sec 
            else:
                # end of beat
                beat = {}
                # number of beat in the bar [1,4]
                beat['num beat'] = beat_num + 1
                # at most one chord per beat
                # check for chords
                new_chord = tune['bars'][bar_num]['beats'][beat_num]['chord']
                beat['chord'] = new_chord
                # pitch of notes which START in this beat
                beat['pitch'] = beat_pitch 
                # duration of notes which START in this beat
                beat['duration'] = beat_duration 
                # offset of notes which START in this beat wrt the start of the bar
                beat['offset'] = []
                # get from chord with m21
                beat['scale'] = []
                beat['bass'] = []
                # append beat
                beats.append(beat)
                beat_pitch = []
                beat_duration = []
                beat_num += 1
                beat_counter += 1
                next_beat_sec = (beat_counter + 1) * beat_duration_sec 
        
        # add note pitch and duration into new_structured_song
        # change chord conditioning from tune based on beat
        # add note to pitch, duration, chord, bass array
        pitch.append(new_pitch)
        duration.append(new_duration)
        chord.append(NottinghamToMidiChords[new_chord][:4])
        bass.append(NottinghamToMidiChords[new_chord][0])
    
    new_structured_song['bars'] = bars
    return new_structured_song

def structuredSongsToPM(structured_song):
    
    # input : a structured song json
    #structured_song = structuredSongs[0]
    #structured_song = new_structured_song
    beat_duration_sec = structured_song['beat duration [sec]']
    tempo = structured_song['tempo']
    
    # sampling of the measure
    unit = beat_duration_sec * 4 / 96.
    # possible note durations in seconds 
    # (it is possible to add representations - include 32nds, quintuplets...):
    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
    possible_durations = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3,
                          unit * 72, unit * 36, unit * 18, unit * 9, 
                          unit * 32, unit * 16, unit * 8, unit * 4]

    # Define durations dictionary
    dur_dict = {}
    dur_dict[possible_durations[0]] = 'full'
    dur_dict[possible_durations[1]] = 'half'
    dur_dict[possible_durations[2]] = 'quarter'
    dur_dict[possible_durations[3]] = '8th'
    dur_dict[possible_durations[4]] = '16th'
    dur_dict[possible_durations[5]] = '32th'
    dur_dict[possible_durations[6]] = 'dot half'
    dur_dict[possible_durations[7]] = 'dot quarter'
    dur_dict[possible_durations[8]] = 'dot 8th'
    dur_dict[possible_durations[9]] = 'dot 16th'
    dur_dict[possible_durations[10]] = 'half note triplet'
    dur_dict[possible_durations[11]] = 'quarter note triplet'
    dur_dict[possible_durations[12]] = '8th note triplet'
    dur_dict[possible_durations[13]] = '16th note triplet'
    inv_dur_dict = {v: k for k, v in dur_dict.items()}
    

    # Construct a PrettyMIDI object.
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Add a piano instrument
    inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    chords_inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    pm.instruments.append(inst)
    pm.instruments.append(chords_inst)
    velocity = 90    
    offset_sec = 0
    beat_counter = 0
    next_beat_sec = (beat_counter + 1) * beat_duration_sec
    last_chord = structured_song['bars'][0]['beats'][0]['chord']
    chord_start = 0
    
    for bar in structured_song['bars']:
        
        if bar['beats'][0]['num beat'] != 1:
            beat_counter += bar['beats'][0]['num beat'] - 1
            offset_sec += (bar['beats'][0]['num beat'] - 1) * beat_duration_sec
            next_beat_sec = (beat_counter + 1) * beat_duration_sec
        
        for beat in bar['beats']:
            
            if beat['chord'] != last_chord:
                # append last chord to pm inst
                if last_chord != 'NC':
                    for chord_pitch in NottinghamToMidiChords[last_chord][:3]:
                        chords_inst.notes.append(pretty_midi.Note(velocity, chord_pitch, chord_start, next_beat_sec - beat_duration_sec))
                # update next chord start
                last_chord = beat['chord']
                chord_start = next_beat_sec - beat_duration_sec

            pitch = beat['pitch']
            duration = beat['duration']
            for i in range(len(pitch)):
                if pitch[i] != '<pad>' and duration[i] != '<pad>':
                    if pitch[i] == 'R':
                        duration_sec = inv_dur_dict[duration[i]]
                    else:
                        duration_sec = inv_dur_dict[duration[i]]
                        start = offset_sec
                        end = offset_sec + duration_sec
                        inst.notes.append(pretty_midi.Note(velocity, pitch[i], start, end))
                    offset_sec += duration_sec
            
            beat_counter += 1
            next_beat_sec = (beat_counter + 1) * beat_duration_sec
    
    # append last chord
    if last_chord != 'NC':
        for chord_pitch in NottinghamToMidiChords[last_chord][:3]:
            chords_inst.notes.append(pretty_midi.Note(velocity, chord_pitch, chord_start, next_beat_sec - beat_duration_sec))
    
    return pm



if __name__ == '__main__':

    # LOAD DATA
    
    NottinghamDB = dataset.NottinghamDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                 con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
    
    #train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
    #val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getValidationData()
    #test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTestData()
    
    songs = NottinghamDB.getOriginalSongDict()
    structuredSongs = NottinghamDB.getStructuredSongs()
    vocabPitch, vocabDuration, vocabBeat = NottinghamDB.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix = NottinghamDB.getInverseVocabs()
    NottinghamChords, NottinghamToMusic21, NottinghamToChordComposition, NottinghamToMidiChords = NottinghamDB.getChordDicts()


    #%% IMPORT PRE-TRAINED MODELS
    
    # PITCH MODEL
    isPitch = True
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 64
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 64
    
    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 32
    bass_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    modelPitch = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, 
                                      beat_vocab_size, beat_embed_dim,  
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx,
                                      device, dropout, isPitch).to(device)
    
    # Import model
    savePATHpitch = 'models/MINGUSpitch_10epochs_seqLen35_NottinghamDB.pt'
    modelPitch.load_state_dict(torch.load(savePATHpitch, map_location=torch.device('cpu')))
    
    
    # DURATION MODEL
    isPitch = False
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 64
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 64
    
    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 32
    bass_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    modelDuration = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, 
                                      beat_vocab_size, beat_embed_dim,  
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx,
                                      device, dropout, isPitch).to(device)
    
    # Import model
    savePATHduration = 'models/MINGUSduration_10epochs_seqLen35_NottinghamDB.pt'
    modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))

    
    #%% GENERATE ON A TUNE
    
    num_bars = 4
    temperature = 1
    new_structured_song = generateCond(structuredSongs[0], num_bars, temperature, 
                                       modelPitch, modelDuration)
    title = new_structured_song['title']
    pm = structuredSongsToPM(new_structured_song)
    pm.write('output/'+ title + '.mid')
    
    
    