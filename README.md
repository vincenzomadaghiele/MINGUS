# MINGUS - Melodic Improvisation Neural Generator Using Seq2seq

##### Supervisor(s): Pasquale Lisena, Raphaël Troncy 
##### Student: Vincenzo Madaghiele

## Project Description

The purpose of this project is to generate a melodic line in the style of a jazz improvisation using machine learning techniques.

A traditional jazz tune has a standard structure: it is composed by a melody, called “theme”, which is usually played at the beginning and at the end of the tune, a “chord progression”, which sets the harmony of the song, and by a series of improvisations by the solo instrumentalists, which compose melodic lines in the moment of the improvisation, taking into account the harmony and the rhythm of the song.

The goal of the developed model should be, therefore, to generate a melodic line starting from a “jazz standard” structure, composed by a “theme” (a melody) and a “harmonic progression” (a series of chords); these information will be the input of the model. The output will be a generated melodic line with style depending on the data used to train the model.

As the project concerns symbolic music generation, the dataset will be composed of jazz improvisation transcriptions in .midi format [1].

In the literature different approaches have been tried to reach similar goals, as for example using Natural Language Processing techniques, for example Jiang, Xia, Berg-Kirkpatrick, “Discovering Music Relations with Sequential Attention” [2] and OpenAI’s MuseNet [3], which both aim at generating melodic lines from a starting melody.

## Running the code

The libraries necessary to running this code are listed in the requirement.txt file. 
After downloading this repository, it is necessary to run the following code in the terminal:
```
$ conda create --name <env> --file requirements.txt
```

## Code description

* MINGUS_dataset_funct.py : functions for data processing
* MINGUS_model.py : functions for model definition, training and validation evaluation 
* MINGUS_train.py : code used for training
* MINGUS_grid_search.py : code for grid search
* MINGUS_eval.py : code for evaluation of the model
* MINGUS_eval_funct.py : functions for model evaluation
* MINGUS_const.py : constants needed for model training
* MINGUS_generate.py : generate over a single sample

## References

[1] https://jazzomat.hfm-weimar.de/dbformat/dboverview.html, https://www.kaggle.com/saikayala/jazz-ml-ready-midi

[2] Junyan Jiang, Gus Xia and Taylor Berg-Kirkpatrick . Discovering Music Relations with Sequential Attention 

[3] https://openai.com/blog/musenet/#fn2

