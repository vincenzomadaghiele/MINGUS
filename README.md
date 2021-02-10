# MINGUS - Melodic Improvisation Neural Generator Using Seq2seq

##### Author: Vincenzo Madaghiele
##### Supervisors: Pasquale Lisena, Raphaël Troncy 


## Project Description

MINGUS is a transformer network for modeling and generation of monophonic jazz melodic lines, named in honour of Charles Mingus (1922 – 1979), American jazz double bassist, composer and pianist.

MINGUS is structured as two parallel transformer encoder models, one predicting pitch and another one predicting duration. This structure was chosen because it allows to capture the rhythmic variation by allowing the model have a lot of different rhythmic values in the dictionary of the duration network.

The purpose of this experiment is to explore the capability of the transformer to model and generate realistic melodic lines in the style of a jazz improvisation. It is also an opportunity to compare the performances of RNN-based models and transformers on musical data.

## Code description

* MINGUS_dataset_funct.py : functions for data processing
* MINGUS_model.py : functions for model definition, training and validation evaluation 
* MINGUS_train.py : code used for training
* MINGUS_grid_search.py : code for grid search
* MINGUS_eval.py : code for evaluation of the model
* MINGUS_eval_funct.py : functions for model evaluation
* MINGUS_const.py : constants needed for model training
* MINGUS_generate.py : generate over a single sample

## Running the code

The libraries necessary to running this code are listed in the requirement.txt file. 
After downloading this repository, run the following code in the terminal to install the dependencies in a conda environment:
```
$ conda create --name <env> --file requirements.txt
```

### Training
To re-train the model run the script MINGUS_train.py. Choose the dataset by typing the name of the folder containing the midi files in line 35. The trained models will be saved in the 'models' folder.

### Evaluation
To evaluate the model run the script MINGUS_eval.py. Choose the dataset by typing the name of the folder containing the midi files in line 52. Choose the trained pitch and duration models from the 'models' folder and type their name in line 106 and 122. The metrics results will be saved as a json file in the 'metrics' folder.

### Generation
To generate on a song run the script MINGUS_generate. Choose the dataset by typing the name of the folder containing the midi files in line 210. Choose the trained pitch and duration models from the 'models' folder and type their name in line 260 and 276. Set the number of notes to generate at line 290. The generated midi file will be saved in the 'output' folder. To generate on multiple songs set line 313 as True and select the dataset in line 316.

