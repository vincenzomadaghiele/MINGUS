#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts pre-processes the input 
musical files in xml or abc formats
for training in MINGUS
"""
import sys
import argparse
import json
import glob
import numpy as np
import music21 as m21
from note_seq import abc_parser

def midiToStructuredSong(midi_path):
    # to complete
    #return new_structured_song
    pass

def abcToStructuredSong(abc_path):
    
    print('Loading abc songbook: ', abc_path)
    
    abcSongbook = abc_parser.parse_abc_tunebook_file(abc_path)
    new_structured_songs = []
    
    # iterate over all valid songs in the abc songbook
    # each song is a note_sequence element 
    for abcSong in abcSongbook[0].values():
        # ensure there is only one time signiture
        # ensure that the song is in 4/4

        time_signatures = abcSong.time_signatures
        tempos = abcSong.tempos
        if len(time_signatures) == 1 and len(tempos) <= 1 and time_signatures[0].numerator == 4 and time_signatures[0].denominator == 4:
    
            # DEFINE THE BASIC SONG PROPERTIES
            metadata = abcSong.sequence_metadata
            
            structured_song = {}
            structured_song['title'] = metadata.title
            structured_song['total time [sec]'] = abcSong.total_time
            structured_song['quantization [sec]'] = abcSong.quantization_info.steps_per_second
            structured_song['quantization [beat]'] = abcSong.quantization_info.steps_per_quarter
            
            print('     Loading song: ', structured_song['title'])
            
            if not abcSong.tempos:
                structured_song['tempo'] = 120
            else:
                structured_song['tempo'] = abcSong.tempos[0]

            beat_duration_sec = 1 / (structured_song['tempo'] / 60)
            
            # sampling of the measure
            unit = beat_duration_sec * 4 / 96.
            # possible note durations in seconds 
            # (it is possible to add representations - include 32nds, quintuplets...):
            # [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]
            possible_durations = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 3,
                                  unit * 72, unit * 36, unit * 18, 
                                  unit * 32, unit * 16]
            
            quarter_durations = [4, 2, 1, 1/2, 1/4, 1/8,
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
            
            chords_times = []
            for textannotation in abcSong.text_annotations:
                if textannotation.annotation_type == 1:
                    chord = textannotation.text
                    chord_time = textannotation.time
                    chords_times.append([chord, chord_time])
            
            # check if the song has chords
            if chords_times:                
                # BASIC ARRAYS FOR structured_songs
                bars = []
                bar_num = 0 # bar count starts from 0 
                beats = []
                # trick: only works for NottinghamDB
                if chords_times[0][1] != 0:
                    beat_num = 3
                    bar_duration = 3
                else:
                    beat_num = 0 # beat count is [0,3]
                    bar_duration = 0
                beat_pitch = []
                beat_duration = []
                beat_offset = []
                offset_sec = 0
                beat_counter = 0
                
                next_beat_sec = (beat_counter + 1) * beat_duration_sec 
                # iterate over the note_sequence notes
                for i in range(len(abcSong.notes)):
                    
                    note = abcSong.notes[i]
                    beat_pitch.append(note.pitch)
                    if note.start_time:
                        duration_sec = note.end_time - note.start_time
                    else:
                        # first note does not have start time
                        duration_sec = note.end_time
                    # calculate distance from each duration
                    distance = np.abs(np.array(possible_durations) - duration_sec)
                    idx = distance.argmin()
                    beat_duration.append(dur_dict[possible_durations[idx]])
                    offset = int(bar_duration * 96 / 4)
                    beat_offset.append(offset)
                    bar_duration += quarter_durations[idx]
                    offset_sec += duration_sec
                    
                    # check for chords
                    nochord = True
                    for j in range(len(chords_times)-1):
                        if chords_times[j+1][1] > note.start_time and chords_times[j][1] <= note.start_time:
                            chord = chords_times[j][0]
                            nochord = False
                        elif chords_times[-1][1] <= note.start_time:
                            chord = chords_times[-1][0]
                            nochord = False
                            break
                    if nochord:
                        print('No chord at song %s, note %d' % (structured_song['title'], i))
                        chord = 'NC'
                    
                    # calculate at which second there is a new beat
                    #next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec + first_note_start
                    while offset_sec >= next_beat_sec:
                        if beat_num >= 3:
                            # end of bar
                            # append beat
                            beat = {}
                            beat['num beat'] = beat_num + 1
                            # check for chords
                            nochord = True
                            for j in range(len(chords_times)-1):
                                if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                    chord = chords_times[j][0]
                                    nochord = False
                                elif chords_times[-1][1] < next_beat_sec:
                                    chord = chords_times[-1][0]
                                    nochord = False
                                    break
                            if nochord:
                                chord = 'NC'
                            beat['chord'] = chord 
                            beat['pitch'] = beat_pitch 
                            beat['duration'] = beat_duration 
                            beat['offset'] = beat_offset
                            beat['scale'] = []
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
                            bar_duration = 0
                        else:
                            # end of beat
                            beat = {}
                            # number of beat in the bar [1,4]
                            beat['num beat'] = beat_num + 1
                            # at most one chord per beat
                            # check for chords
                            nochord = True
                            for j in range(len(chords_times)-1):
                                if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                    chord = chords_times[j][0]
                                    nochord = False
                                elif chords_times[-1][1] < next_beat_sec:
                                    chord = chords_times[-1][0]
                                    nochord = False
                                    break
                            if nochord:
                                chord = 'NC'
                            beat['chord'] = chord 
                            # pitch of notes which START in this beat
                            beat['pitch'] = beat_pitch 
                            # duration of notes which START in this beat
                            beat['duration'] = beat_duration 
                            # offset of notes which START in this beat wrt the start of the bar
                            beat['offset'] = beat_offset
                            # get from chord with m21
                            beat['scale'] = []
                            beat['bass'] = []
                            # append beat
                            beats.append(beat)
                            beat_pitch = []
                            beat_duration = []
                            beat_offset = []
                            beat_num += 1
                            beat_counter += 1
                            next_beat_sec = (beat_counter + 1) * beat_duration_sec 

                    
                    # check for rests
                    
                    if i != len(abcSong.notes)-1:
                        intra_note_time = abcSong.notes[i+1].start_time - abcSong.notes[i].end_time
                        # if the interval between notes is greater than the smallest duration ('16th')
                        # and smaller than the greatest duration ('full') then there is a rest
                        if intra_note_time >= possible_durations[5]:
                            # there is a rest!
                            # handle the possibility of pauses longer than a full note
                            while intra_note_time > possible_durations[0]:
                                
                                beat_pitch.append('R')
                                # calculate distance from each duration
                                distance = np.abs(np.array(possible_durations) - intra_note_time)
                                idx = distance.argmin()
                                beat_duration.append(dur_dict[possible_durations[idx]])
                                offset = int(bar_duration * 96 / 4)
                                beat_offset.append(offset)
                                bar_duration += quarter_durations[idx]

                                offset_sec += duration_sec
                                intra_note_time -= possible_durations[idx]
                                
                                # check for chords
                                nochord = True
                                for j in range(len(chords_times)-1):
                                    if chords_times[j+1][1] > note.start_time and chords_times[j][1] <= note.start_time:
                                        chord = chords_times[j][0]
                                        nochord = False
                                    elif chords_times[-1][1] <= note.start_time:
                                        chord = chords_times[-1][0]
                                        nochord = False
                                        break
                                if nochord:
                                    print('No chord at song %s, note %d' % (structured_song['title'], i))
                                    chord = 'NC'
                                
                                # calculate at which second there is a new beat
                                #next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec 
                                while offset_sec >= next_beat_sec:
                                    if beat_num >= 3:
                                        # end of bar
                                        # append beat
                                        beat = {}
                                        beat['num beat'] = beat_num + 1
                                        # check for chords
                                        nochord = True
                                        for j in range(len(chords_times)-1):
                                            if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                                chord = chords_times[j][0]
                                                nochord = False
                                            elif chords_times[-1][1] < next_beat_sec:
                                                chord = chords_times[-1][0]
                                                nochord = False
                                                break
                                        if nochord:
                                            chord = 'NC'
                                        beat['chord'] = chord 
                                        beat['pitch'] = beat_pitch 
                                        beat['duration'] = beat_duration 
                                        beat['offset'] = beat_offset
                                        beat['scale'] = []
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
                                    else:
                                        # end of beat
                                        beat = {}
                                        # number of beat in the bar [1,4]
                                        beat['num beat'] = beat_num + 1
                                        # at most one chord per beat
                                        # check for chords
                                        nochord = True
                                        for j in range(len(chords_times)-1):
                                            if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                                chord = chords_times[j][0]
                                                nochord = False
                                            elif chords_times[-1][1] < next_beat_sec:
                                                chord = chords_times[-1][0]
                                                nochord = False
                                                break
                                        if nochord:
                                            chord = 'NC'
                                        beat['chord'] = chord 
                                        # pitch of notes which START in this beat
                                        beat['pitch'] = beat_pitch 
                                        # duration of notes which START in this beat
                                        beat['duration'] = beat_duration 
                                        # offset of notes which START in this beat wrt the start of the bar
                                        beat['offset'] = beat_offset
                                        # get from chord with m21
                                        beat['scale'] = []
                                        beat['bass'] = []
                                        # append beat
                                        beats.append(beat)
                                        beat_pitch = []
                                        beat_duration = []
                                        beat_offset = []
                                        beat_num += 1
                                        beat_counter += 1
                                        next_beat_sec = (beat_counter + 1) * beat_duration_sec 
                            
                            beat_pitch.append('R')
                            # calculate distance from each duration
                            distance = np.abs(np.array(possible_durations) - intra_note_time)
                            idx = distance.argmin()
                            beat_duration.append(dur_dict[possible_durations[idx]])
                            offset = int(bar_duration * 96 / 4)
                            beat_offset.append(offset)
                            bar_duration += quarter_durations[idx]
                            offset_sec += duration_sec
                            # check for chords
                            nochord = True
                            for j in range(len(chords_times)-1):
                                if chords_times[j+1][1] > note.start_time and chords_times[j][1] <= note.start_time:
                                    chord = chords_times[j][0]
                                    nochord = False
                                elif chords_times[-1][1] <= note.start_time:
                                    chord = chords_times[-1][0]
                                    nochord = False
                                    break
                            if nochord:
                                print('No chord at song %s, note %d' % (structured_song['title'], i))
                                chord = 'NC'
                            
                            # calculate at which second there is a new beat
                            #next_beat_sec = (bar_num * 4 + beat_num + 1) * beat_duration_sec 
                            while offset_sec >= next_beat_sec:
                                if beat_num >= 3:
                                    # end of bar
                                    # append beat
                                    beat = {}
                                    beat['num beat'] = beat_num + 1
                                    # check for chords
                                    nochord = True
                                    for j in range(len(chords_times)-1):
                                        if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                            chord = chords_times[j][0]
                                            nochord = False
                                        elif chords_times[-1][1] < next_beat_sec:
                                            chord = chords_times[-1][0]
                                            nochord = False
                                            break
                                    if nochord:
                                        chord = 'NC'
                                    beat['chord'] = chord 
                                    beat['pitch'] = beat_pitch 
                                    beat['duration'] = beat_duration 
                                    beat['offset'] = beat_offset
                                    beat['scale'] = []
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
                                else:
                                    # end of beat
                                    beat = {}
                                    # number of beat in the bar [1,4]
                                    beat['num beat'] = beat_num + 1
                                    # at most one chord per beat
                                    # check for chords
                                    nochord = True
                                    for j in range(len(chords_times)-1):
                                        if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                                            chord = chords_times[j][0]
                                            nochord = False
                                        elif chords_times[-1][1] < next_beat_sec:
                                            chord = chords_times[-1][0]
                                            nochord = False
                                            break
                                    if nochord:
                                        chord = 'NC'
                                    beat['chord'] = chord 
                                    # pitch of notes which START in this beat
                                    beat['pitch'] = beat_pitch 
                                    # duration of notes which START in this beat
                                    beat['duration'] = beat_duration 
                                    # offset of notes which START in this beat wrt the start of the bar
                                    beat['offset'] = beat_offset
                                    # get from chord with m21
                                    beat['scale'] = []
                                    beat['bass'] = []
                                    # append beat
                                    beats.append(beat)
                                    beat_pitch = []
                                    beat_duration = []
                                    beat_offset = []
                                    beat_num += 1
                                    beat_counter += 1
                                    next_beat_sec = (beat_counter + 1) * beat_duration_sec 
                            
                            
                # append last bar in case it doesn't have 4 beats
                if beats:
                    # end of bar
                    # append beat
                    beat = {}
                    beat['num beat'] = beat_num + 1
                    # check for chords
                    nochord = True
                    for j in range(len(chords_times)-1):
                        if chords_times[j+1][1] >= next_beat_sec and chords_times[j][1] < next_beat_sec:
                            chord = chords_times[j][0]
                            nochord = False
                        elif chords_times[-1][1] < next_beat_sec:
                            chord = chords_times[-1][0]
                            nochord = False
                            break
                    if nochord:
                        chord = 'NC'
                    beat['chord'] = chord 
                    beat['pitch'] = beat_pitch 
                    beat['duration'] = beat_duration 
                    beat['offset'] = beat_offset
                    beat['scale'] = []
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
                                    
                structured_song['bars'] = bars
                structured_song['beat duration [sec]'] = beat_duration_sec
                
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
            
                structured_song['bars'] = bars
                #songs.append(song)
                new_structured_songs.append(structured_song)

    return new_structured_songs

def xmlToStructuredSong(xml_path):
    
    print('Loading song: ', xml_path)
    
    '''
    # Import xml file
    possible_durations = [4, 2, 1, 1/2, 1/4, 1/8, 1/16,
                          3, 3/2, 3/4, 3/8, 
                          1/6, 1/12]
    
    min_rest = 1/16

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
    
    min_rest = 1/2

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
    bass = 'R'

    for measure in s.getElementsByClass('Measure'):
        #measure.show('text')
        bar_num += 1
        beat_num = 0
        bar_duration = 0
        for note in measure.notesAndRests:
            if 'Rest' in note.classSet:
                # detect rests
                pitch = 'R'
                if note.quarterLength > min_rest:
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
                chord = note.figure
                pitchNames = [str(p) for p in note.pitches]
                if len(pitchNames) < 4:
                    hd = m21.harmony.ChordStepModification('add', 7)
                    note.addChordStepModification(hd, updatePitches=True)
                    pitchNames = [str(p) for p in note.pitches]
                # midi conversion
                midiChord = []
                for p in pitchNames:
                    c = m21.pitch.Pitch(p)
                    midiChord.append(c.midi)
                # define bass
                bass = midiChord[0]
            
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
                    new_beat['bass'] = bass
                    new_beat['this beat duration [sec]'] = new_structured_song['beat duration [sec]']
                    beats.append(new_beat)
                    beat_num += 1
                    
                if not chord:
                    chord = 'NC'
                    bass = 'R'
                new_beat = {}
                new_beat['num beat'] = int(beat_num) + 1
                new_beat['chord'] = chord
                new_beat['pitch'] = beat_pitch 
                new_beat['duration'] = beat_duration 
                new_beat['offset'] = beat_offset
                new_beat['scale'] = []
                new_beat['bass'] = bass
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

    return new_structured_song

def arraysFromStructuredSong(structuredTune):
    
    # exclude short rests
    rests_durations = ['full','half','quarter','8th',
                       'dot half', 'dot quarter', 'dot 8th',
                       'half note triplet']
    
    song = {}
    song['title'] = structuredTune['title']
    song['avgtempo'] = structuredTune['tempo']
    song['beat duration [sec]'] = structuredTune['beat duration [sec]']

    pitch = []
    duration = []
    chord = []
    next_chord = []
    bass = []
    beat = []
    offset = []
    for bar in structuredTune['bars']:
        for bea in bar['beats']:
            # exclude long rests
            for p in range(len(bea['pitch'])):
                if bea['pitch'][p] == 'R':
                    if bea['duration'][p] in rests_durations:
                        pitch.append(bea['pitch'][p])
                        duration.append(bea['duration'][p])
                        offset.append(min(bea['offset'][p],96))
                        if not bea['bass']:
                            bea['bass'] = 'R'
                        if bea['bass'] != 'R':
                            if np.isnan(bea['bass']):
                                bea['bass'] = 'R'
        
                        chord.append(bea['chord'])
                        next_chord.append(bea['next chord'])
                        bass.append(bea['bass'])
                        beat.append(min(bea['num beat'],4))
                else:
                    pitch.append(int(bea['pitch'][p]))
                    duration.append(bea['duration'][p])
                    offset.append(min(bea['offset'][p],96))
                    if not bea['bass']:
                        bea['bass'] = 'R'
                    if bea['bass'] != 'R':
                        if np.isnan(bea['bass']):
                            bea['bass'] = 'R'

                    chord.append(bea['chord'])
                    next_chord.append(bea['next chord'])
                    bass.append(bea['bass'])
                    beat.append(min(bea['num beat'],4))
            '''
            for p in bea['pitch']:
                if p == 'R':
                    pitch.append(p)
                else:
                    pitch.append(int(p))
                chord.append(bea['chord'])
                next_chord.append(bea['next chord'])
                if not bea['bass']:
                    bea['bass'] = 'R'
                if bea['bass'] != 'R':
                    if np.isnan(bea['bass']):
                        bea['bass'] = 'R'
                bass.append(bea['bass'])
                beat.append(min(bea['num beat'],4))
            for d in bea['duration']:
                duration.append(d)
            for o in bea['offset']:
                offset.append(min(o,96))
            '''
    song['pitch'] = pitch
    song['duration'] = duration
    song['chords'] = chord
    song['next chords'] = next_chord
    song['bass pitch'] = bass
    song['beats'] = beat
    song['offset'] = offset
    
    return song


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, default='xml',
                        help='data format for preprocessing')
    args = parser.parse_args(sys.argv[1:])
    
    
    structuredSongs = []
    songs = []
    if args.format == 'xml':
        print('Loading dataset from xml: ')
        print('-' * 80)
        source_path = 'A_preprocessData/data/xml/*.xml'
        source_songs = glob.glob(source_path)
        for xml_path in source_songs:
            structuredTune = xmlToStructuredSong(xml_path)
            tune = arraysFromStructuredSong(structuredTune)
            structuredSongs.append(structuredTune)
            songs.append(tune)

    elif args.format == 'abc':
        print('Loading dataset from abc: ')
        print('-' * 80)
        source_path = 'A_preprocessData/data/abc/*.abc'
        source_songs = glob.glob(source_path)
        for abc_path in source_songs:
            structuredTunes = abcToStructuredSong(abc_path)
            for structuredTune in structuredTunes:
                tune = arraysFromStructuredSong(structuredTune)
                structuredSongs.append(structuredTune)
                songs.append(tune)
    
    
    # split into train, validation and test
    songs_split = {}
    # train: 70% 
    songs_split['train'] = songs[:int(len(songs)*0.7)]
    # train: 10% 
    songs_split['validation'] = songs[int(len(songs)*0.7)+1:int(len(songs)*0.7)+1+int(len(songs)*0.1)]
    # train: 20%
    songs_split['test'] = songs[int(len(songs)*0.7)+1+int(len(songs)*0.1):]
    # structured songs (ordered by bar and beats)
    songs_split['structured for generation'] = structuredSongs
    
    # Convert dict to JSON and SAVE IT
    with open('A_preprocessData/data/DATA.json', 'w') as fp:
        json.dump(songs_split, fp, indent=4)

