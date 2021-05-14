#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:16:31 2021

@author: vincenzomadaghiele

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
            for p in bea['pitch']:
                if p == 'R':
                    pitch.append(p)
                else:
                    pitch.append(int(p))
                chord.append(bea['chord'])
                next_chord.append(bea['next chord'])
                bass.append(bea['bass'])
                beat.append(min(bea['num beat'],4))
            for d in bea['duration']:
                duration.append(d)
            for o in bea['offset']:
                offset.append(o)
    
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
        print('abc')
    
    
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

