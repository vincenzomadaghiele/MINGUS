#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:01:13 2021

@author: vincenzomadaghiele
"""


# Constants for MINGUS training

TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
# length of one note sequence (use powers of 2 for even divisions)
BPTT = 35 
AUGMENTATION = True
SEGMENTATION = True
augmentation_const = 5