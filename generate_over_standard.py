#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:05:49 2021

@author: vincenzomadaghiele

This scripts generates over a jazz standard given in XML format as input to the model
"""

import music21 as m21
import json
import glob
import numpy as np
import torch
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con
import MINGUS_condGenerate as gen
import CustomDB_dataPrep as cus

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

def xmlToStructuredSong(xml_path, datasetToMusic21,
                        datasetToMidiChords, datasetToChordComposition, datasetChords):
    '''
    # Import xml file
    possible_durations = [4, 2, 1, 1/2, 1/4, 1/8, 1/16,
                          3, 3/2, 3/4, 3/8, 
                          1/6, 1/12]

    # Define durations dictionary
    dur_dict = {}
    dur_dict[possible_durations[0]] = 'full'
    dur_dict[possible_durations[1]] = 'half'
    dur_dict[possible_durations[2]] = 'quarter'
    dur_dict[possible_durations[3]] = '8th'
    dur_dict[possible_durations[4]] = '16th'
    dur_dict[possible_durations[5]] = '32th'
    dur_dict[possible_durations[6]] = '64th'
    dur_dict[possible_durations[7]] = 'dot half'
    dur_dict[possible_durations[8]] = 'dot quarter'
    dur_dict[possible_durations[9]] = 'dot 8th'
    dur_dict[possible_durations[10]] = 'dot 16th'
    dur_dict[possible_durations[11]] = 'half note triplet'
    dur_dict[possible_durations[12]] = 'quarter note triplet'
    '''
    
    # Import xml file
    possible_durations = [4, 2, 1, 1/2, 1/4, 1/8,
                          3, 3/2, 3/4,
                          1/6, 1/12]
    
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
    dur_dict[possible_durations[9]] = 'half note triplet'
    dur_dict[possible_durations[10]] = 'quarter note triplet'    
    
    # invert dict from Wjazz to Music21 chords
    Music21ToWjazz = {v: k for k, v in datasetToMusic21.items()}
    
    s = m21.converter.parse(xml_path)
    
    new_structured_song = {}
    new_structured_song['title'] = xml_path[22:-4]
    new_structured_song['tempo'] = s.metronomeMarkBoundaries()[0][2].number
    new_structured_song['beat duration [sec]'] = 60 / new_structured_song['tempo']
    
    if not s.hasMeasures():
        s = s.makeMeasures()
    #s.show('text')
    bar_num = 0
    bars = []
    beats = []
    beat_pitch = []
    beat_duration = []
    beat_offset = []
    chord = 'NC'

    for measure in s.getElementsByClass('Measure'):
        #measure.show('text')
        bar_num += 1
        beat_num = 0
        bar_duration = 0
        for note in measure.notesAndRests:
            if 'Rest' in note.classSet:
                # detect rests
                pitch = 'R'
                distance = np.abs(np.array(possible_durations) - note.quarterLength)
                idx = distance.argmin()
                duration = dur_dict[possible_durations[idx]]
                offset = int(bar_duration * 96 / 4)
                # update beat arrays 
                beat_pitch.append(pitch)
                beat_duration.append(duration)
                beat_offset.append(offset)            
            
            elif 'ChordSymbol' in note.classSet:
                #chord
                m21chord = note.figure
                if m21chord in Music21ToWjazz.keys():
                    chord = Music21ToWjazz[m21chord]
                else:
                    # add to WjazzToMusic21
                    datasetToMusic21[m21chord] = m21chord
                    # derive chord composition and make it of 4 notes
                    pitchNames = [str(p) for p in note.pitches]
                    # The following bit is added 
                    # just for parameter modeling purposes
                    if len(pitchNames) < 4:
                        hd = m21.harmony.ChordStepModification('add', 7)
                        note.addChordStepModification(hd, updatePitches=True)
                        #chord = m21.chord.Chord(pitchNames)
                        pitchNames = [str(p) for p in note.pitches]   
                    
                    # midi conversion
                    midiChord = []
                    for p in pitchNames:
                        c = m21.pitch.Pitch(p)
                        midiChord.append(c.midi)
                    
                    chord = m21chord
                    NewChord = {}
                    NewChord['Wjazz name'] = chord
                    NewChord['music21 name'] = m21chord
                    NewChord['chord composition'] = pitchNames
                    NewChord['midi chord composition'] = midiChord
                    NewChord['one-hot encoding'] = []
                    WjazzChords.append(NewChord)

                    # update dictionaries
                    WjazzToMidiChords[chord] = midiChord[:4]
                    WjazzToChordComposition[chord] = pitchNames[:4]
            
            # check for rests
            else:
                pitch = note.pitch.midi
                distance = np.abs(np.array(possible_durations) - note.quarterLength)
                idx = distance.argmin()
                duration = dur_dict[possible_durations[idx]]
                offset = int(bar_duration * 96 / 4)
                # update beat arrays 
                beat_pitch.append(pitch)
                beat_duration.append(duration)
                beat_offset.append(offset)

            # update bar duration
            bar_duration += note.quarterLength
            # check if the beat is ended
            if np.floor(bar_duration) != beat_num:
                
                #print(np.floor(bar_duration), beat_num)
                while np.floor(bar_duration) - beat_num > 1:
                    new_beat = {}
                    new_beat['num beat'] = int(beat_num) + 1
                    new_beat['chord'] = chord
                    new_beat['pitch'] = [] 
                    new_beat['duration'] = [] 
                    new_beat['offset'] = []
                    new_beat['scale'] = []
                    new_beat['bass'] = WjazzToMidiChords[chord][0]
                    new_beat['this beat duration [sec]'] = new_structured_song['beat duration [sec]']
                    beats.append(new_beat)
                    beat_num += 1
                    
                if not chord:
                    chord = 'NC'
                new_beat = {}
                new_beat['num beat'] = int(beat_num) + 1
                new_beat['chord'] = chord
                new_beat['pitch'] = beat_pitch 
                new_beat['duration'] = beat_duration 
                new_beat['offset'] = beat_offset
                new_beat['scale'] = []
                new_beat['bass'] = WjazzToMidiChords[chord][0]
                new_beat['this beat duration [sec]'] = new_structured_song['beat duration [sec]']
                beats.append(new_beat)
                if beat_num == 3:
                    # append bar
                    new_bar = {}
                    new_bar['num bar'] = bar_num # over all song
                    new_bar['beats'] = beats # beats 1,2,3,4
                    bars.append(new_bar)
                    beats = []
                
                beat_num = np.floor(bar_duration)
                beat_pitch = []
                beat_duration = []
                beat_offset = []
    
    # compute chords array
    chord_array = []
    for bar in bars:
        for beat in bar['beats']:
            chord_array.append(beat['chord'])
    
    # compute next chord 
    last_chord = chord_array[0]
    next_chords = []
    for i in range(len(chord_array)):
        if chord_array[i] != last_chord:
            next_chords.append(chord_array[i])
            last_chord = chord_array[i]
    
    # compute array of next chords
    next_chords.append('NC')
    next_chord_array = []
    next_chord_pointer = 0
    last_chord = chord_array[0]
    for i in range(len(chord_array)):
        if chord_array[i] != last_chord:
            last_chord = chord_array[i]
            next_chord_pointer += 1
        next_chord_array.append(next_chords[next_chord_pointer])
    
    
    # compute next chord 
    last_chord = bars[0]['beats'][0]['chord']
    next_chords2 = []
    for bar in bars:
        for beat in bar['beats']:
            if beat['chord'] != last_chord:
                next_chords2.append(beat['chord'])
                last_chord = beat['chord']
    
    # add next chord to the beats
    last_chord = bars[0]['beats'][0]['chord']
    next_chords2.append('NC')
    next_chord_pointer = 0
    for bar in bars:
        for beat in bar['beats']:
            if beat['chord'] != last_chord:
                last_chord = beat['chord']
                next_chord_pointer += 1
            beat['next chord'] = next_chords2[next_chord_pointer]

    new_structured_song['bars'] = bars    

    return new_structured_song, datasetToMusic21, datasetToMidiChords, datasetToChordComposition, datasetChords


def generateOverStandard(tune, num_chorus, temperature,
                         modelPitch, modelDuration, 
                         datasetToMidiChords, 
                         pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                         vocabPitch, vocabDuration,
                         isJazz = False):
    
    # copy bars into a new_structured_song
    new_structured_song = {}
    new_structured_song['title'] = tune['title'] + '_gen'
    new_structured_song['tempo'] = tune['tempo']
    new_structured_song['beat duration [sec]'] = tune['beat duration [sec]']
    beat_duration_sec = new_structured_song['beat duration [sec]'] 
    bars = []
    beats = []
    
    print('Generating over song %s' % (tune['title']))
    
    '''
    # sampling of the measure
    unit = beat_duration_sec * 4 / 192.
    # possible note durations in seconds 
    # (it is possible to add representations - include 32nds, quintuplets...):
    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
    possible_durations = [unit * 192, unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3, 
                          unit * 144, unit * 72, unit * 36, unit * 18, 
                          unit * 32, unit * 16]

    # Define durations dictionary
    dur_dict = {}
    dur_dict[possible_durations[0]] = 'full'
    dur_dict[possible_durations[1]] = 'half'
    dur_dict[possible_durations[2]] = 'quarter'
    dur_dict[possible_durations[3]] = '8th'
    dur_dict[possible_durations[4]] = '16th'
    dur_dict[possible_durations[5]] = '32th'
    dur_dict[possible_durations[6]] = '64th'
    dur_dict[possible_durations[7]] = 'dot half'
    dur_dict[possible_durations[8]] = 'dot quarter'
    dur_dict[possible_durations[9]] = 'dot 8th'
    dur_dict[possible_durations[10]] = 'dot 16th'
    dur_dict[possible_durations[11]] = 'half note triplet'
    dur_dict[possible_durations[12]] = 'quarter note triplet'
    inv_dur_dict = {v: k for k, v in dur_dict.items()}
    '''
    
    # sampling of the measure
    unit = beat_duration_sec * 4 / 96.
    # possible note durations in seconds 
    # (it is possible to add representations - include 32nds, quintuplets...):
    # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
    possible_durations = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3, 
                          unit * 72, unit * 36, unit * 18, 
                          unit * 16, unit * 8]

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
    dur_dict[possible_durations[9]] = 'half note triplet'
    dur_dict[possible_durations[10]] = 'quarter note triplet'
    inv_dur_dict = {v: k for k, v in dur_dict.items()}
    
    # initialize counters
    bar_num = len(tune['bars'])
    chorus_len = len(tune['bars'])
    this_chorus_num = 1
    beat_counter = 0
    offset_sec = 0
    beat_num = 0
    bar_onset = 0
    for bar in tune['bars']:
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
    while len(bars) < len(tune['bars']) * num_chorus :
        
        # only give last 128 characters to speed up generation
        last_char = -128
        #print(len(pitch[last_char:]))
        
        # batchify
        pitch4gen = gen.batch4gen(pitch[last_char:], len(pitch[last_char:]), pitch_to_ix, device)
        duration4gen = gen.batch4gen(duration[last_char:], len(duration[last_char:]), duration_to_ix, device)
        chord4gen = gen.batch4gen(chord[last_char:], len(chord[last_char:]), pitch_to_ix, device, isChord=True)
        next_chord4gen = gen.batch4gen(chord[last_char:], len(next_chord[last_char:]), pitch_to_ix, device, isChord=True)
        bass4gen = gen.batch4gen(bass[last_char:], len(bass[last_char:]), pitch_to_ix, device)
        beat4gen = gen.batch4gen(beat_ar[last_char:], len(beat_ar[last_char:]), beat_to_ix, device)
        offset4gen = gen.batch4gen(offset[last_char:], len(offset[last_char:]), offset_to_ix, device)
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
                    if len(tune['bars']) * num_chorus > bar_num:
                        new_chord = tune['bars'][bar_num - (chorus_len * this_chorus_num)]['beats'][beat_num]['chord']
                        new_next_chord = tune['bars'][bar_num - (chorus_len * this_chorus_num)]['beats'][beat_num]['next chord']
                    beat['chord'] = new_chord
                    beat['pitch'] = beat_pitch 
                    beat['duration'] = beat_duration 
                    beat['offset'] = beat_offset
                    beat['scale'] = []
                    if isJazz:
                        if len(tune['bars']) * num_chorus > bar_num:
                            new_bass = tune['bars'][bar_num - (chorus_len * this_chorus_num)]['beats'][beat_num]['bass']
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
                    
                    if bar_num % chorus_len == 0:
                        this_chorus_num += 1
                        
                else:
                    # end of beat
                    beat = {}
                    # number of beat in the bar [1,4]
                    beat['num beat'] = beat_num + 1
                    # at most one chord per beat
                    # check for chords
                    if len(tune['bars']) * num_chorus > bar_num:
                        new_chord = tune['bars'][bar_num - (chorus_len * this_chorus_num)]['beats'][beat_num]['chord']
                        new_next_chord = tune['bars'][bar_num - (chorus_len * this_chorus_num)]['beats'][beat_num]['next chord']
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
                        if len(tune['bars']) * num_chorus > bar_num:
                            new_bass = tune['bars'][bar_num - (chorus_len * this_chorus_num)]['beats'][beat_num]['bass']
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

    elif con.DATASET == 'CustomDB':
        
        CustomDB = dataset.CustomDB(device, con.TRAIN_BATCH_SIZE, con.EVAL_BATCH_SIZE,
                                    con.BPTT, con.AUGMENTATION, con.SEGMENTATION, con.augmentation_const)
            
        #train_pitch_batched, train_duration_batched, train_chord_batched, train_next_chord_batched, train_bass_batched, train_beat_batched, train_offset_batched  = CustomDB.getTrainingData()
        #val_pitch_batched, val_duration_batched, val_chord_batched, val_next_chord_batched, val_bass_batched, val_beat_batched, val_offset_batched  = CustomDB.getValidationData()
        #test_pitch_batched, test_duration_batched, test_chord_batched, test_next_chord_batched, test_bass_batched, test_beat_batched, test_offset_batched  = CustomDB.getTestData()

        songs = CustomDB.getOriginalSongDict()
        structuredSongs = CustomDB.getStructuredSongs()
        vocabPitch, vocabDuration, vocabBeat, vocabOffset = CustomDB.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = CustomDB.getInverseVocabs()
        WjazzChords, WjazzToMusic21, WjazzToChordComposition, WjazzToMidiChords = CustomDB.getChordDicts()


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
        savePATHpitch = 'models/MINGUSpitch_10epochs_seqLen35_WjazzDB.pt'
        
        savePATHpitch = f'models/{con.DATASET}/pitchModel/MINGUS COND {con.COND_TYPE_PITCH} Epochs {con.EPOCHS}.pt'
        
    elif con.DATASET == 'NottinghamDB':
        savePATHpitch = 'models/MINGUSpitch_100epochs_seqLen35_NottinghamDB.pt'
    
    elif con.DATASET == 'CustomDB':
        savePATHpitch = f'models/{con.DATASET}/pitchModel/MINGUS COND {con.COND_TYPE_PITCH} Epochs {con.EPOCHS}.pt'
    
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
        savePATHduration = 'models/MINGUSduration_10epochs_seqLen35_WjazzDB.pt'
        
        savePATHduration = f'models/{con.DATASET}/durationModel/MINGUS COND {con.COND_TYPE_DURATION} Epochs {con.EPOCHS}.pt'
        
    elif con.DATASET == 'NottinghamDB':
        savePATHduration = 'models/MINGUSduration_100epochs_seqLen35_NottinghamDB.pt'
    
    elif con.DATASET == 'CustomDB':
        savePATHduration = f'models/{con.DATASET}/durationModel/MINGUS COND {con.COND_TYPE_DURATION} Epochs {con.EPOCHS}.pt'
    
    modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))


    #%% Import xml file
    
    xml_path = 'output/tunes4eval/xml/All_The_Things_You_Are_short.xml'    
    tuneFromXML, WjazzToMusic21, WjazzToMidiChords, WjazzToChordComposition, WjazzChords = xmlToStructuredSong(xml_path, 
                                                                                                               WjazzToMusic21,
                                                                                                               WjazzToMidiChords, 
                                                                                                               WjazzToChordComposition, 
                                                                                                               WjazzChords)
    
    #tuneFromXML = cus.xmlToStructuredSong(xml_path)

    # GENERATE ON A TUNE
    num_chorus = 4
    temperature = 1
    
    isJazz = False
    new_structured_song = generateOverStandard(tuneFromXML, num_chorus, temperature, 
                                               modelPitch, modelDuration, WjazzToMidiChords, 
                                               pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                               vocabPitch, vocabDuration,
                                               isJazz)
    title = new_structured_song['title']
    pm = gen.structuredSongsToPM(new_structured_song, WjazzToMidiChords, isJazz)
    pm.write('output/' + title + '.mid')

    
    #%% BUILD A DATASET OF GENERATED TUNES
    
    # Set to True to generate dataset of songs
    generate_dataset = True
    
    if generate_dataset:
        out_path = 'output/gen4eval_WjazzDB/'
        generated_path = 'generated/'
        original_path = 'original/'
        num_tunes = 20
        generated_structuredSongs = []
        original_structuredSongs = []


        source_path = 'output/tunes4eval/xml/*.xml'
        source_songs = glob.glob(source_path)
        for xml_path in source_songs:
            
            if con.DATASET == 'WjazzDB':
                isJazz = True
                tune, WjazzToMusic21, WjazzToMidiChords, WjazzToChordComposition, WjazzChords = xmlToStructuredSong(xml_path, 
                                                                                                                   WjazzToMusic21,
                                                                                                                   WjazzToMidiChords, 
                                                                                                                   WjazzToChordComposition, 
                                                                                                                   WjazzChords)

                
                new_structured_song = generateOverStandard(tune, num_chorus, temperature, 
                                                       modelPitch, modelDuration, WjazzToMidiChords, 
                                                       pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                                       vocabPitch, vocabDuration,
                                                       isJazz)
                
                pm = gen.structuredSongsToPM(new_structured_song, WjazzToMidiChords, isJazz)
                pm.write(out_path + generated_path + new_structured_song['title'] + '.mid')
                pm = gen.structuredSongsToPM(tune, WjazzToMidiChords, isJazz)
                pm.write(out_path + original_path + tune['title'] + '.mid')
                generated_structuredSongs.append(new_structured_song)
                original_structuredSongs.append(tune)
                
            elif con.DATASET == 'NottinghamDB':
                isJazz = False
                tune, WjazzToMusic21, WjazzToMidiChords, WjazzToChordComposition, WjazzChords = xmlToStructuredSong(xml_path, 
                                                                                                                   WjazzToMusic21,
                                                                                                                   WjazzToMidiChords, 
                                                                                                                   WjazzToChordComposition, 
                                                                                                                   WjazzChords)
                new_structured_song = generateOverStandard(tune, num_chorus, temperature, 
                                                       modelPitch, modelDuration, NottinghamToMidiChords,
                                                       pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                                       vocabPitch, vocabDuration)
                
                pm = gen.structuredSongsToPM(new_structured_song, NottinghamToMidiChords)
                pm.write(out_path + generated_path + new_structured_song['title'] + '.mid')
                pm = gen.structuredSongsToPM(tune, NottinghamToMidiChords)
                pm.write(out_path + original_path + tune['title'] + '.mid')
                generated_structuredSongs.append(new_structured_song)
                original_structuredSongs.append(tune)
                
            if con.DATASET == 'CustomDB':
                isJazz = False
                tune, WjazzToMusic21, WjazzToMidiChords, WjazzToChordComposition, WjazzChords = xmlToStructuredSong(xml_path, 
                                                                                                                   WjazzToMusic21,
                                                                                                                   WjazzToMidiChords, 
                                                                                                                   WjazzToChordComposition, 
                                                                                                                   WjazzChords)

                
                new_structured_song = generateOverStandard(tune, num_chorus, temperature, 
                                                       modelPitch, modelDuration, WjazzToMidiChords, 
                                                       pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                                                       vocabPitch, vocabDuration,
                                                       isJazz)
                
                pm = gen.structuredSongsToPM(new_structured_song, WjazzToMidiChords, isJazz)
                pm.write('output/00_MINGUS_gens/' + new_structured_song['title'] + '.mid')
                generated_structuredSongs.append(new_structured_song)
                
        
        if con.DATASET != 'CustomDB':
            # Convert dict to JSON and SAVE IT
            with open(out_path + generated_path + con.DATASET + '_generated.json', 'w') as fp:
                json.dump(generated_structuredSongs, fp, indent=4)
            with open(out_path + original_path + con.DATASET + '_original.json', 'w') as fp:
                json.dump(original_structuredSongs, fp, indent=4)

        else:
            # convert reference to midi and structured song json
            source_path = 'output/reference/xml/*.xml'
            source_songs = glob.glob(source_path)
            reference_structuredSongs = []
            for xml_path in source_songs:
                tune = cus.xmlToStructuredSong(xml_path) # remove min_rest
                reference_structuredSongs.append(tune)
                pm = gen.structuredSongsToPM(tune, WjazzToMidiChords)
                pm.write('output/reference/' + tune['title'] + '.mid')
            
            with open('output/reference/'+ con.DATASET + '_reference.json', 'w') as fp:
                json.dump(reference_structuredSongs, fp, indent=4)
            with open('output/00_MINGUS_gens/'+ con.DATASET + '_generated.json', 'w') as fp:
                json.dump(generated_structuredSongs, fp, indent=4)



    
