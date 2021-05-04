#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:09:31 2021

@author: vincenzomadaghiele
"""
import glob
import json
import CustomDB_dataPrep as cus

if __name__ == '__main__':
    
    structuredSongs = []
    songs = []
    source_path = 'output/01_BebopNet_gens/*.xml'
    source_songs = glob.glob(source_path)
    for xml_path in source_songs:
        structuredTune = cus.xmlToStructuredSong(xml_path)
        structuredSongs.append(structuredTune)
        
    # Convert dict to JSON and SAVE IT
    with open('output/01_BebopNet_gens/CustomDB_generated.json', 'w') as fp:
        json.dump(structuredSongs, fp, indent=4)
