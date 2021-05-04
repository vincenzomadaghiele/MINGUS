#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:01:13 2021

@author: vincenzomadaghiele
"""

# Constants for MINGUS training

TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10

# length of one note sequence

BPTT = 35
AUGMENTATION = False
SEGMENTATION = False
augmentation_const = 2

DATASET = 'CustomDB'

EPOCHS = 50

COND_TYPE_PITCH = 'I-C-NC-BE-O'
COND_TYPE_DURATION = 'I-C-NC-BE-O' 

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