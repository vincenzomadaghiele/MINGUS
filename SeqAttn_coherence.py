#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:04:35 2021

@author: vincenzomadaghiele
"""

import music21 as m21 
import pretty_midi
import glob

if __name__ == '__main__':

    source_path = 'output/02_SeqAttn_gens_NottinghamDB/*.mid'
    source_songs = glob.glob(source_path)
    scale_coherence = 0
    chord_coherence = 0
    count_pitch = 0

    for file in source_songs:
        
        #file = 'output/02_SeqAttn_gens/0_given_0bars_generate_64bars.mid'
        pm = pretty_midi.PrettyMIDI(file)
        print("Loading Music File:", file)
        
        try:
            melody = pm.instruments[0]
            chords = pm.instruments[1]   
        
            last_start = 0
            chord = []
            chords_list = []
            for note in range(len(chords.notes)-1):
                if chords.notes[note].start == last_start:
                    chord.append(chords.notes[note].pitch)
                else:
                    chords_list.append((chord, last_start))
                    chord = []
                    last_start = chords.notes[note].start
                    
                    
            for note in range(len(melody.notes)-1):
                for k in range(len(chords_list)):
                    if k < len(chords_list)-1:
                        if chords_list[k][1] <= melody.notes[note].start and chords_list[k+1][1] > melody.notes[note].start:
                            current_chord = chords_list[k][0]
                            break
                
                if len(current_chord) != 0:
                    m21chord = m21.harmony.chordSymbolFigureFromChord(m21.chord.Chord(current_chord), True)[0]
                    try:
                        h = m21.harmony.ChordSymbol(m21chord)
                        hd = m21.harmony.ChordStepModification('add', 2)
                        h.addChordStepModification(hd, updatePitches=True)
                        hd = m21.harmony.ChordStepModification('add', 3)
                        h.addChordStepModification(hd, updatePitches=True)
                        hd = m21.harmony.ChordStepModification('add', 4)
                        h.addChordStepModification(hd, updatePitches=True)
                        hd = m21.harmony.ChordStepModification('add', 5)
                        h.addChordStepModification(hd, updatePitches=True)
                        hd = m21.harmony.ChordStepModification('add', 6)
                        h.addChordStepModification(hd, updatePitches=True)
                        hd = m21.harmony.ChordStepModification('add', 7)
                        h.addChordStepModification(hd, updatePitches=True)
                        hd = m21.harmony.ChordStepModification('add', 8)
                        h.addChordStepModification(hd, updatePitches=True)
                        scale = [m21.pitch.Pitch(pitch).name for pitch in h.pitches]
                        chordPitch = [m21.pitch.Pitch(midiPitch).name for midiPitch in current_chord]
                
                        pitchName = m21.pitch.Pitch(melody.notes[note].pitch).name
                        if pitchName in scale:
                            scale_coherence += 1
                        if pitchName in chordPitch:
                            chord_coherence += 1
                        count_pitch += 1
                    except:
                        print('Could not convert chord:',m21chord)
        except:
            print('Could not convert song:', file)

    scale_coherence = scale_coherence / count_pitch
    chord_coherence = chord_coherence / count_pitch

