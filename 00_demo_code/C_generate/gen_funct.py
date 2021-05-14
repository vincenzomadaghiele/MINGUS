#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:40:35 2021

@author: vincenzomadaghiele
"""
import torch
import pretty_midi
import numpy as np
import music21 as m21


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
    new_structured_song['title'] = xml_path.split('/')[-1].split('.')[0]
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
                    datasetChords.append(NewChord)

                    # update dictionaries
                    datasetToMidiChords[chord] = midiChord[:4]
                    datasetToChordComposition[chord] = pitchNames[:4]
            
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
                    new_beat['bass'] = datasetToMidiChords[chord][0]
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
                new_beat['bass'] = datasetToMidiChords[chord][0]
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


def generateOverStandard(tune, num_chorus, temperature,
                         modelPitch, modelDuration, 
                         datasetToMidiChords, 
                         pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
                         vocabPitch, vocabDuration,
                         BPTT, device,
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
        src_mask_pitch = modelPitch.generate_square_subsequent_mask(BPTT).to(device)
        src_mask_duration = modelDuration.generate_square_subsequent_mask(BPTT).to(device)
        if pitch4gen.size(0) != BPTT:
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

def structuredSongsToPM(structured_song, datasetToMidiChords, isJazz = False):
    
    # input : a structured song json
    #structured_song = structuredSongs[0]
    #structured_song = new_structured_song
    beat_duration_sec = structured_song['beat duration [sec]']
    tempo = structured_song['tempo']
    
    
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
                if last_chord != 'NC' and last_chord in datasetToMidiChords.keys():
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
    if last_chord != 'NC' and last_chord in datasetToMidiChords.keys():
        for chord_pitch in datasetToMidiChords[last_chord][:3]:
            chords_inst.notes.append(pretty_midi.Note(velocity, int(chord_pitch), chord_start, next_beat_sec - beat_duration_sec))
    if isJazz:
        if last_bass != 'R':
            bass_inst.notes.append(pretty_midi.Note(bass_velocity, int(last_bass), bass_start, next_beat_sec - beat_duration_sec))
       
    return pm

