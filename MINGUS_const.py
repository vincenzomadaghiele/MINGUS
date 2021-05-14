#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:01:13 2021

@author: vincenzomadaghiele

ToDO:
    - COMPUTE ALL MGEVAL METRICS AND UPDATE PAPER
    - explain tables in supplementary material
    - move params to yml
    - substitute chord dicts with extractChord function with m21
    - move gen funct to separate py file
    - remove duplicate chordxml function
    - check that augmentation and no segmentation work in the code
    - adapt input to abc
    
ToDO (later):
    - insert SPECIAL pre-processing functions for WjazzDB and Nottingham 
        alternative to data_preprocessing.py which generate a DATA.json 
        (with bass and all)
    
ToASK:
    - is the new code ok?
    - do we have to include trained models in the folders?
    - there is no BASS in the code we give (no WjazzDB) !!!
    - do I have to include ablation study and conversion?
    - do I have to include WjazzDB and NottinghamDB data prep?
    - should I implement early stopping? 
"""

# Constants for MINGUS training

TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10

# length of one note sequence
BPTT = 35
AUGMENTATION = False
SEGMENTATION = True
augmentation_const = 2

DATASET = 'CustomDB'

EPOCHS = 100

COND_TYPE_PITCH = 'I-C-B-BE-O'
COND_TYPE_DURATION = 'B-BE-O'

'''
COND TYPES:
    - NO : no conditioning
    - I : inter conditioning
    - C : chord conditioning
    - NC : next chord conditioning
    - B : bass conditioning
    - BE : beat conditioning
    - O : offset conditioning
    - COMBINE : I-C-NC-B-BE-O
    
Best to the ear so far:
    COND_TYPE_PITCH = 'I-C-NC-BE'
    COND_TYPE_DURATION = 'I-B-BE-O'
'''