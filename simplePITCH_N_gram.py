#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:41:11 2020

@author: vincenzomadaghiele

from: pytorch embedding tutorial
"""

# library for understanding music
from music21 import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

EMBEDDING_DIM = 10
CONTEXT_SIZE = 32
    

# define the Dataset object for Pytorch
class ImprovDataset(Dataset):
    
    """
    --> DataLoader can do the batch computation for us

    Implement a custom Dataset:
    inherit Dataset
    implement __init__ , __getitem__ , and __len__
    """
    
    def __init__(self):
        #for listing down the file names
        import os
        
        #specify the path
        path='data/w_jazz/'
        #read all the filenames
        files=[i for i in os.listdir(path) if i.endswith(".mid")]
        #reading each midi file
        notes_array = np.array([read_midi(path+i) for i in files])
        
        #converting 2D array into 1D array
        notes_ = [element for note_ in notes_array for element in note_]
        #No. of unique notes
        unique_notes = list(set(notes_))
        print("number of unique notes: " + str(len(unique_notes)))
        
        from collections import Counter
        #computing frequency of each note
        freq = dict(Counter(notes_))
        
        # CLARIFY WHY THIS HAS TO BE DONE (do not exclude notes in the future)
        
        # the threshold for frequent notes can change 
        threshold=50 # this threshold is the number of classes that have to be predicted
        frequent_notes = [note_ for note_, count in freq.items() if count>=threshold]
        print("number of frequent notes (more than 50 times): " + str(len(frequent_notes)))
        self.num_frequent_notes = len(frequent_notes)
        
        # prepare new musical files which contain only the top frequent notes
        new_music=[]
        for notes in notes_array:
            temp=[]
            for note_ in notes:
                if note_ in frequent_notes:
                    temp.append(note_)            
            new_music.append(temp)
        new_music = np.array(new_music) # same solos but with only most frequent notes
        
        # Preparing sequences for the model prediction 
        no_of_timesteps = 32 # the solos are divided up into sequences of 32 notes
        x = []
        y = []
        for note_ in new_music:
            for i in range(0, len(note_) - no_of_timesteps, 1):        
                #preparing input and output sequences
                input_ = note_[i:i + no_of_timesteps] # sequences with step 1
                output = note_[i + no_of_timesteps] # prediction
                
                x.append(input_)
                y.append(output)
        
        x=np.array(x)
        y=np.array(y)

        self.x = x
        self.y = y

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def getData(self):
        return self.x, self.y

#defining function to read MIDI files
def read_midi(file):
    
    print("Loading Music File:",file)
    
    notes=[]
    notes_to_parse = None
    
    #parsing a midi file
    midi = converter.parse(file)
  
    #grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    #Looping over all the instruments
    for part in s2.parts:
        notes_to_parse = part.recurse() 
        #finding whether a particular element is note or a chord
        for element in notes_to_parse:        
            #note
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))        
            #chord
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    return np.array(notes)

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


if __name__ == '__main__':

    # LOAD DATASET
    dataset = ImprovDataset()
    X, y = dataset.getData()
    vocab = set(X.ravel())
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    X = [sequence.tolist() for sequence in X]
    
    # Format the data for embedding
    samples = [(X[i],y[i]) for i in range(len(y))]
    print(samples[:3])

    # Train embeddings
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # Trainging loop
    for epoch in range(10):
        total_loss = 0
        for context, target in samples:
    
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
    
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            model.zero_grad()
    
            # Step 3. Run the forward pass, getting log probabilities over next words
            log_probs = model(context_idxs)
    
            # Step 4. Compute your loss function. (Again, Torch wants the targetword wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
    
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
    
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)  # The loss decreased every iteration over the training data!




    