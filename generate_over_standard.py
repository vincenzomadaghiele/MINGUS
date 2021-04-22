#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:05:49 2021

@author: vincenzomadaghiele

This scripts generates over a jazz standard given in XML format as input to the model
"""
import pretty_midi
import music21 as m21
import json
import numpy as np
import torch
import torch.nn as nn
import loadDBs as dataset
import MINGUS_condModel as mod
import MINGUS_const as con
import MINGUS_condGenerate as gen

def xmlToStructuredSong(xml_path, datasetToMusic21,
                        datasetToMidiChords, datasetToChordComposition, datasetChords):
    
    # Import xml file
    possible_durations = [4, 2, 1, 1/2, 1/4, 1/8,
                          3, 1 + 1/2, 1/2 + 1/4, 1/4 + 1/8, 
                          4/3]

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
    
    # invert dict from Wjazz to Music21 chords
    Music21ToWjazz = {v: k for k, v in datasetToMusic21.items()}
    
    s = m21.converter.parse(xml_path)
    
    new_structured_song = {}
    new_structured_song['title'] = xml_path[20:-4]
    new_structured_song['tempo'] = s.metronomeMarkBoundaries()[0][2].number
    new_structured_song['beat duration [sec]'] = 60 / new_structured_song['tempo']
    
    if not s.hasMeasures():
        s = s.makeMeasures()
    #sMeasures.show('text')
    bar_num = 0
    bars = []
    beats = []
    beat_pitch = []
    beat_duration = []
    beat_offset = []

    for measure in s.getElementsByClass('Measure'):
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
                count = np.floor(bar_duration) - beat_num
                while count > 1:
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
                    count -= 1
                    
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
                    # if there are missing beats add them
                    count = len(beats)
                    while count < 4:
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
                        count = len(beats)
                        
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
    
    new_structured_song['bars'] = bars
    
    return new_structured_song, datasetToMusic21, datasetToMidiChords, datasetToChordComposition, datasetChords

def generateOverStandard(tune, model, num_chorus):
    # upload a standard in XML format
    # xml to structured song
    # use chord root as bass
    # extract chord sequence
    # generate over number of chorus chosen
    # export in midi and xml
    #return stream
    pass

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


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
        savePATHpitch = 'models/MINGUSpitch_10epochs_seqLen35_WjazzDB.pt'
        
        #savePATHpitch = f'models/{con.DATASET}/pitchModel/MINGUS COND {con.COND_TYPE_PITCH} Epochs {con.EPOCHS}.pt'
        
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
        savePATHduration = 'models/MINGUSduration_10epochs_seqLen35_WjazzDB.pt'
        
        #savePATHduration = f'models/{con.DATASET}/durationModel/MINGUS COND {con.COND_TYPE_DURATION} Epochs {con.EPOCHS}.pt'
        
    elif con.DATASET == 'NottinghamDB':
        savePATHduration = 'models/MINGUSduration_100epochs_seqLen35_NottinghamDB.pt'
    modelDuration.load_state_dict(torch.load(savePATHduration, map_location=torch.device('cpu')))


    #%% Import xml file
    
    xml_path = 'output/xmlStandards/Giant_Steps.xml'
    
    '''
    possible_durations = [4, 2, 1, 1/2, 1/4, 1/8,
                          3, 1 + 1/2, 1/2 + 1/4, 1/4 + 1/8, 
                          4/3]

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
    
    # invert dict from Wjazz to Music21 chords
    Music21ToWjazz = {v: k for k, v in WjazzToMusic21.items()}
    
    s = m21.converter.parse('output/xmlStandards/Never_Gonna_Give_You_Up.xml')
    
    new_structured_song = {}
    new_structured_song['title'] = 'Never_Gonna_Give_You_Up'
    new_structured_song['tempo'] = s.metronomeMarkBoundaries()[0][2].number
    new_structured_song['beat duration [sec]'] = 60 / new_structured_song['tempo']
    
    if not s.hasMeasures():
        s = s.makeMeasures()
    #sMeasures.show('text')
    bar_num = 0
    bars = []
    beats = []
    beat_pitch = []
    beat_duration = []
    beat_offset = []

    for measure in s.getElementsByClass('Measure'):
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
                    WjazzToMusic21[m21chord] = m21chord
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
                count = np.floor(bar_duration) - beat_num
                while count > 1:
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
                    count -= 1
                    
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
                    # if there are missing beats add them
                    count = len(beats)
                    while count < 4:
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
                        count = len(beats)
                        
                    # append bar
                    new_bar = {}
                    new_bar['num bar'] = bar_num # over all song
                    new_bar['beats'] = beats # beats 1,2,3,4
                    bars.append(new_bar)
                    beats = []
                    bar_onset = 0
                
                beat_num = np.floor(bar_duration)
                beat_pitch = []
                beat_duration = []
                beat_offset = []
    
    new_structured_song['bars'] = bars
    '''
    
    new_structured_song, WjazzToMusic21, WjazzToMidiChords, WjazzToChordComposition, WjazzChords = xmlToStructuredSong(xml_path, WjazzToMusic21,
                                                                                                                       WjazzToMidiChords, WjazzToChordComposition, WjazzChords)
    
    #%% Convert structured song to midi
    
    title = new_structured_song['title'] + '_standard'
    pm = gen.structuredSongsToPM(new_structured_song, WjazzToMidiChords, isJazz = True)
    pm.write('output/'+ title + '.mid')

    
