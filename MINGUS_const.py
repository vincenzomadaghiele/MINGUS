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
SEGMENTATION = True
augmentation_const = 4

DATASET = 'WjazzDB'

EPOCHS = 200