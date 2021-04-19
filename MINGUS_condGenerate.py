#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 07:09:14 2021

@author: vincenzomadaghiele
"""

import pretty_midi
import json
import numpy as np
import torch
import torch.nn as nn
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

def generateCond(tune, num_bars, temperature, 
                 modelPitch, modelDuration, 
                 datasetToMidiChords, 
                 pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                 vocabPitch, vocabDuration,
                 isJazz = False):
        
    # select a tune
    #tune = structuredSongs[0]
    # choose how many starting bars
    #num_bars = 4
    # define temperature
    #temperature = 1   
    
    # copy bars into a new_structured_song
    new_structured_song = {}
    new_structured_song['title'] = tune['title'] + '_gen'
    new_structured_song['tempo'] = tune['tempo']
    new_structured_song['beat duration [sec]'] = tune['beat duration [sec]']
    beat_duration_sec = new_structured_song['beat duration [sec]'] 
    if isJazz:
        new_structured_song['chord changes'] = tune['chord changes']
    bars = []
    beats = []
    
    print('Generating over song %s' % (tune['title']))
    
    '''
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
    '''
    
    # sampling of the measure
    unit = beat_duration_sec * 4 / 96.
    # possible note durations in seconds 
    # (it is possible to add representations - include 32nds, quintuplets...):
    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
    possible_durations = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3,
                          unit * 72, unit * 36, unit * 18, unit * 9, 
                          unit * 32]

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
    inv_dur_dict = {v: k for k, v in dur_dict.items()}
    
    
    # initialize counters
    bar_num = num_bars
    beat_counter = 0
    offset_sec = 0
    beat_num = 0
    bar_onset = 0
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
    beat_offset = []
    next_beat_sec = (beat_counter + 1) * beat_duration_sec 
    
    # extract starting bars pitch, duration, chord, bass and put into array
    pitch = []
    duration = []
    chord = []
    next_chord = []
    bass = []
    beat_ar = []
    offset = []
    for bar in bars:
        for beat in bar['beats']:
            for i in range(len(beat['pitch'])):
                pitch.append(beat['pitch'][i])
                duration.append(beat['duration'][i])
                chord.append(datasetToMidiChords[beat['chord']][:4])
                next_chord.append(datasetToMidiChords[beat['next chord']][:4])
                if isJazz:
                    bass.append(beat['bass'])
                else:
                    bass.append(datasetToMidiChords[beat['chord']][0])
                beat_ar.append(beat['num beat'])
                offset.append(beat['offset'][i])
        
    new_chord = beat['chord']
    new_next_chord = beat['next chord']
    if isJazz:
        new_bass = beat['bass']
    while len(bars) < len(tune['bars']):
        
        # only give last 128 characters to speed up generation
        last_char = -128
        #print(len(pitch[last_char:]))
        
        # batchify
        pitch4gen = batch4gen(pitch[last_char:], len(pitch[last_char:]), pitch_to_ix, device)
        duration4gen = batch4gen(duration[last_char:], len(duration[last_char:]), duration_to_ix, device)
        chord4gen = batch4gen(chord[last_char:], len(chord[last_char:]), pitch_to_ix, device, isChord=True)
        next_chord4gen = batch4gen(chord[last_char:], len(next_chord[last_char:]), pitch_to_ix, device, isChord=True)
        bass4gen = batch4gen(bass[last_char:], len(bass[last_char:]), pitch_to_ix, device)
        beat4gen = batch4gen(beat_ar[last_char:], len(beat_ar[last_char:]), beat_to_ix, device)
        offset4gen = batch4gen(offset[last_char:], len(offset[last_char:]), offset_to_ix, device)
        # reshape to column vectors
        pitch4gen = pitch4gen.t()
        duration4gen = duration4gen.t()
        chord4gen = chord4gen.t()
        chord4gen = chord4gen.reshape(chord4gen.shape[0], 1, chord4gen.shape[1])
        next_chord4gen = next_chord4gen.t()
        next_chord4gen = next_chord4gen.reshape(next_chord4gen.shape[0], 1, next_chord4gen.shape[1])
        bass4gen = bass4gen.t()
        beat4gen = beat4gen.t()
        offset4gen = offset4gen.t()
        
        # generate new note conditioning on old arrays
        modelPitch.eval()
        modelDuration.eval()
        src_mask_pitch = modelPitch.generate_square_subsequent_mask(con.BPTT).to(device)
        src_mask_duration = modelDuration.generate_square_subsequent_mask(con.BPTT).to(device)
        if pitch4gen.size(0) != con.BPTT:
            src_mask_pitch = modelPitch.generate_square_subsequent_mask(pitch4gen.size(0)).to(device)
            src_mask_duration = modelDuration.generate_square_subsequent_mask(duration4gen.size(0)).to(device)
        
        # generate new pitch note
        pitch_pred = modelPitch(pitch4gen, duration4gen, chord4gen, next_chord4gen,
                                bass4gen, beat4gen, offset4gen, src_mask_pitch)
        word_weights = pitch_pred[-1].squeeze().div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0].item()
        new_pitch = vocabPitch[word_idx]
        # generate new duration note
        duration_pred = modelDuration(pitch4gen, duration4gen, chord4gen, next_chord4gen,
                                   bass4gen, beat4gen, offset4gen, src_mask_duration)
        word_weights = duration_pred[-1].squeeze().div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0].item()
        new_duration = vocabDuration[word_idx]
        
        
        # if any padding is generated jump the step and re-try (it is rare!)
        if new_pitch != '<pad>' and new_duration != '<pad>':
            # append note to new arrays
            beat_pitch.append(new_pitch)
            beat_duration.append(new_duration)
            
            duration_sec = inv_dur_dict[new_duration] # problem: might generate padding here
            offset_sec += duration_sec
            
            new_offset = min(int(bar_onset / (beat_duration_sec * 4) * 96),96)
            bar_onset += duration_sec
            
            beat_offset.append(new_offset)
            
            # check if the note is in a new beat / bar
            while offset_sec >= next_beat_sec:
                if beat_num >= 3:
                    # end of bar
                    # append beat
                    beat = {}
                    beat['num beat'] = beat_num + 1
                    # check for chords
                    if len(tune['bars']) > bar_num:
                        new_chord = tune['bars'][bar_num]['beats'][beat_num]['chord']
                        new_next_chord = tune['bars'][bar_num]['beats'][beat_num]['next chord']
                    beat['chord'] = new_chord
                    beat['pitch'] = beat_pitch 
                    beat['duration'] = beat_duration 
                    beat['offset'] = beat_offset
                    beat['scale'] = []
                    if isJazz:
                        if len(tune['bars']) > bar_num:
                            new_bass = tune['bars'][bar_num]['beats'][beat_num]['bass']
                        beat['bass'] = new_bass
                    else:
                        beat['bass'] = []
                    beats.append(beat)
                    beat_pitch = []
                    beat_duration = []
                    beat_offset = []
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
                    bar_onset = 0
                else:
                    # end of beat
                    beat = {}
                    # number of beat in the bar [1,4]
                    beat['num beat'] = beat_num + 1
                    # at most one chord per beat
                    # check for chords
                    if len(tune['bars']) > bar_num:
                        new_chord = tune['bars'][bar_num]['beats'][beat_num]['chord']
                        new_next_chord = tune['bars'][bar_num]['beats'][beat_num]['next chord']
                    beat['chord'] = new_chord
                    # pitch of notes which START in this beat
                    beat['pitch'] = beat_pitch 
                    # duration of notes which START in this beat
                    beat['duration'] = beat_duration 
                    # offset of notes which START in this beat wrt the start of the bar
                    beat['offset'] = beat_offset
                    # get from chord with m21
                    beat['scale'] = []
                    if isJazz:
                        if len(tune['bars']) > bar_num:
                            new_bass = tune['bars'][bar_num]['beats'][beat_num]['bass']
                        beat['bass'] = new_bass
                    else:
                        beat['bass'] = []
                    # append beat
                    beats.append(beat)
                    beat_pitch = []
                    beat_duration = []
                    beat_offset = []
                    beat_num += 1
                    beat_counter += 1
                    next_beat_sec = (beat_counter + 1) * beat_duration_sec 
            
            # add note pitch and duration into new_structured_song
            # change chord conditioning from tune based on beat
            # add note to pitch, duration, chord, bass array
            pitch.append(new_pitch)
            duration.append(new_duration)
            chord.append(datasetToMidiChords[new_chord][:4])
            next_chord.append(datasetToMidiChords[new_next_chord][:4])
            beat_ar.append(beat_num + 1)
            if isJazz:
                bass.append(new_bass)
            else:
                bass.append(datasetToMidiChords[new_chord][0])
            offset.append(new_offset)
    
    new_structured_song['bars'] = bars
    return new_structured_song

def structuredSongsToPM(structured_song, datasetToMidiChords):
    
    # input : a structured song json
    #structured_song = structuredSongs[0]
    #structured_song = new_structured_song
    beat_duration_sec = structured_song['beat duration [sec]']
    tempo = structured_song['tempo']
    
    '''
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
    '''
    
    # sampling of the measure
    unit = beat_duration_sec * 4 / 96.
    # possible note durations in seconds 
    # (it is possible to add representations - include 32nds, quintuplets...):
    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
    possible_durations = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3,
                          unit * 72, unit * 36, unit * 18, unit * 9, 
                          unit * 32]

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
    inv_dur_dict = {v: k for k, v in dur_dict.items()}


    # Construct a PrettyMIDI object.
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Add a piano instrument
    inst = pretty_midi.Instrument(program=67, is_drum=False, name='Tenor Sax')
    chords_inst = pretty_midi.Instrument(program=1, is_drum=False, name='piano')
    pm.instruments.append(inst)
    pm.instruments.append(chords_inst)
    if isJazz:
        bass_inst = pretty_midi.Instrument(program=44, is_drum=False, name='Contrabass')
        pm.instruments.append(bass_inst)
    velocity = 95
    chord_velocity = 60
    bass_velocity = 85
    offset_sec = 0
    beat_counter = 0
    next_beat_sec = (beat_counter + 1) * beat_duration_sec
    last_chord = structured_song['bars'][0]['beats'][0]['chord']
    chord_start = 0
    if isJazz:
        last_bass = structured_song['bars'][0]['beats'][0]['bass']
        bass_start = 0
    
    for bar in structured_song['bars']:
        
        if bar['beats'][0]['num beat'] != 1:
            beat_counter += bar['beats'][0]['num beat'] - 1
            offset_sec += (bar['beats'][0]['num beat'] - 1) * beat_duration_sec
            next_beat_sec = (beat_counter + 1) * beat_duration_sec
        
        for beat in bar['beats']:
            
            if beat['chord'] != last_chord:
                # append last chord to pm inst
                if last_chord != 'NC':
                    for chord_pitch in datasetToMidiChords[last_chord][:3]:
                        chords_inst.notes.append(pretty_midi.Note(chord_velocity, int(chord_pitch), chord_start, next_beat_sec - beat_duration_sec))
                # update next chord start
                last_chord = beat['chord']
                chord_start = next_beat_sec - beat_duration_sec

            if isJazz:
                if beat['bass'] != last_bass:
                    # append last chord to pm inst
                    if last_bass != 'R': 
                        bass_inst.notes.append(pretty_midi.Note(bass_velocity, int(last_bass), bass_start, next_beat_sec - beat_duration_sec))
                    # update next chord start
                    last_bass = beat['bass']
                    bass_start = next_beat_sec - beat_duration_sec

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
                        inst.notes.append(pretty_midi.Note(chord_velocity, int(pitch[i]), start, end))
                    offset_sec += duration_sec
            
            beat_counter += 1
            next_beat_sec = (beat_counter + 1) * beat_duration_sec
    
    # append last chord
    if last_chord != 'NC':
        for chord_pitch in datasetToMidiChords[last_chord][:3]:
            chords_inst.notes.append(pretty_midi.Note(velocity, int(chord_pitch), chord_start, next_beat_sec - beat_duration_sec))
    if isJazz:
        if last_bass != 'R':
            bass_inst.notes.append(pretty_midi.Note(bass_velocity, int(last_bass), bass_start, next_beat_sec - beat_duration_sec))
       
    return pm



if __name__ == '__main__':

    # LOAD DATA
    if con.DATASET == 'WjazzDB':
            
        WjazzDB = dataset.WjazzDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                     con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
        
        #train_pitch_batched, train_duration_batched, train_chord_batched, train_bass_batched, train_beat_batched  = WjazzDB.getTrainingData()
        #val_pitch_batched, val_duration_batched, val_chord_batched, val_bass_batched, val_beat_batched  = WjazzDB.getValidationData()
        #test_pitch_batched, test_duration_batched, test_chord_batched, test_bass_batched, test_beat_batched  = WjazzDB.getTestData()
        
        songs = WjazzDB.getOriginalSongDict()
        structuredSongs = WjazzDB.getStructuredSongs()
        vocabPitch, vocabDuration, vocabBeat, vocabOffset = WjazzDB.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = WjazzDB.getInverseVocabs()
        WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = WjazzDB.getChordDicts()

    elif con.DATASET == 'NottinghamDB':
        
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


    #%% LOAD PRE-TRAINED MODELS
    
    # PITCH MODEL
    isPitch = True
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 512
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 512
    
    chord_encod_dim = 64
    next_chord_encod_dim = 32

    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 64
    bass_embed_dim = 64
    
    offset_vocab_size = len(vocabOffset) # size of the duration vocabulary
    offset_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    offset_pad_idx = offset_to_ix['<pad>']
    modelPitch = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
                                      beat_vocab_size, beat_embed_dim,
                                      offset_vocab_size, offset_embed_dim,
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
                                      device, dropout, isPitch, con.COND_TYPE_PITCH).to(device)
    
    if con.DATASET == 'WjazzDB':
        savePATHpitch = 'models/MINGUSpitch_100epochs_seqLen35_WjazzDB.pt'
        
        savePATHpitch = f'models/{con.DATASET}/pitchModel/MINGUS COND {con.COND_TYPE_PITCH} Epochs {con.EPOCHS}.pt'
        
    elif con.DATASET == 'NottinghamDB':
        savePATHpitch = 'models/MINGUSpitch_100epochs_seqLen35_NottinghamDB.pt'
    modelPitch.load_state_dict(torch.load(savePATHpitch, map_location=torch.device('cpu')))
    
    
    # DURATION MODEL
    isPitch = False
    pitch_vocab_size = len(vocabPitch) # size of the pitch vocabulary
    pitch_embed_dim = 64
    
    duration_vocab_size = len(vocabDuration) # size of the duration vocabulary
    duration_embed_dim = 64
    
    chord_encod_dim = 64
    next_chord_encod_dim = 32
    
    beat_vocab_size = len(vocabBeat) # size of the duration vocabulary
    beat_embed_dim = 32
    bass_embed_dim = 32
    
    offset_vocab_size = len(vocabOffset) # size of the duration vocabulary
    offset_embed_dim = 32


    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    offset_pad_idx = offset_to_ix['<pad>']
    modelDuration = mod.TransformerModel(pitch_vocab_size, pitch_embed_dim,
                                      duration_vocab_size, duration_embed_dim, 
                                      bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
                                      beat_vocab_size, beat_embed_dim, 
                                      offset_vocab_size, offset_embed_dim,
                                      emsize, nhead, nhid, nlayers, 
                                      pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
                                      device, dropout, isPitch, con.COND_TYPE_DURATION).to(device)
    
    if con.DATASET == 'WjazzDB':
        savePATHduration = 'models/MINGUSduration_100epochs_seqLen35_WjazzDB.pt'
        
        savePATHduration = f'models/{con.DATASET}/durationModel/MINGUS COND {con.COND_TYPE_DURATION} Epochs {con.EPOCHS}.pt'
        
    elif con.DATASET == 'NottinghamDB':
        savePATHduration = 'models/MINGUSduration_100epochs_seqLen35_NottinghamDB.pt'
    modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))

    
    #%% GENERATE ON A TUNE
    
    num_bars = 8
    temperature = 1
    
    if con.DATASET == 'WjazzDB':
        isJazz = True
        new_structured_song = generateCond(structuredSongs[0], num_bars, temperature, 
                                       modelPitch, modelDuration, WjazzToMidiChords, 
                                       pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                       vocabPitch, vocabDuration,
                                       isJazz)
        title = new_structured_song['title']
        pm = structuredSongsToPM(new_structured_song, WjazzToMidiChords)
        pm.write('output/'+ title + '.mid')
    elif con.DATASET == 'NottinghamDB':
        isJazz = False
        new_structured_song = generateCond(structuredSongs[0], num_bars, temperature, 
                                       modelPitch, modelDuration, NottinghamToMidiChords,
                                       pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                       vocabPitch, vocabDuration)
        title = new_structured_song['title']
        pm = structuredSongsToPM(new_structured_song, NottinghamToMidiChords)
        pm.write('output/'+ title + '.mid')
    
    
    #%% BUILD A DATASET OF GENERATED TUNES
    
    # Set to True to generate dataset of songs
    generate_dataset = True
    
    if generate_dataset:
        out_path = 'output/gen4eval_' + con.DATASET + '/'
        generated_path = 'generated/'
        original_path = 'original/'
        num_tunes = 20
        generated_structuredSongs = []
        original_structuredSongs = []

        for tune in structuredSongs[:num_tunes]:
            
            if con.DATASET == 'WjazzDB':
                isJazz = True
                new_structured_song = generateCond(tune, num_bars, temperature, 
                                           modelPitch, modelDuration, WjazzToMidiChords, 
                                           pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                           vocabPitch, vocabDuration,
                                           isJazz)
                
                pm = structuredSongsToPM(new_structured_song, WjazzToMidiChords)
                pm.write(out_path + generated_path + new_structured_song['title'] + '.mid')
                pm = structuredSongsToPM(tune, WjazzToMidiChords)
                pm.write(out_path + original_path + tune['title'] + '.mid')
                generated_structuredSongs.append(new_structured_song)
                original_structuredSongs.append(tune)
                
            elif con.DATASET == 'NottinghamDB':
                isJazz = False
                new_structured_song = generateCond(tune, num_bars, temperature, 
                                           modelPitch, modelDuration, NottinghamToMidiChords,
                                           pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                           vocabPitch, vocabDuration)
                
                pm = structuredSongsToPM(new_structured_song, NottinghamToMidiChords)
                pm.write(out_path + generated_path + new_structured_song['title'] + '.mid')
                pm = structuredSongsToPM(tune, NottinghamToMidiChords)
                pm.write(out_path + original_path + tune['title'] + '.mid')
                generated_structuredSongs.append(new_structured_song)
                original_structuredSongs.append(tune)
            
        # Convert dict to JSON and SAVE IT
        with open(out_path + generated_path + con.DATASET + '_generated.json', 'w') as fp:
            json.dump(generated_structuredSongs, fp, indent=4)
        with open(out_path + original_path + con.DATASET + '_original.json', 'w') as fp:
            json.dump(original_structuredSongs, fp, indent=4)

    