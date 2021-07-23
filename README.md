# MINGUS - Melodic Improvisation Neural Generator Using Seq2seq

This is the official repository of the MINGUS project. It contains all the code for the paper:
> MINGUS: Melodic Improvisation Neural Generator Using Seq2Seq
>
> Vincenzo Madaghiele, Pasquale Lisena, Raphaël Troncy
>
> EURECOM, Sophia Antipolis, France  
>
> Published in Proceedings of 22nd International Society of Music Information Retrieval Conference, ISMIR 2021

The web application for user evaluation is available at [mingus.tools.eurecom.fr](https://mingus.tools.eurecom.fr/).
Supplementary material with additional information on the paper is available [here](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf).
Jazz melodies generated by MINGUS are available at [here](https://github.com/vincenzomadaghiele/MINGUS/tree/master/E_docs/melodies) 

### Abstract:
Sequence to Sequence (Seq2Seq) approaches have shown good performances in automatic music generation. We introduce MINGUS, a Transformer-based Seq2Seq architecture for modelling and generating monophonic jazz melodic lines. MINGUS relies on two dedicated embedding models (respectively for pitch and duration) and exploits in prediction features such as chords (current and following), bass line, position inside the measure. The obtained results are comparable with the state of the art, with particularly good performances on jazz music.

## 0. Install dependencies
To use the model run the following code from the main project directory.
After downloading this repository, run the following code in the terminal to install the required libraries in a new conda environment named MINGUS and activate it:
```
$ conda env create -f environment.yml
$ conda activate MINGUS
```
If this installation does not work correctly you can install the dependencies listed in the file requirement.txt individually.
It will probably be necessary to run the following command:
```
$ export PYTHONPATH="$PWD"
```

## 1. Pre-process data
To train MINGUS with a custom dataset place your musicXML data in 00_preprocessData/data/xml and run the following command in your terminal:
```
$ python3 A_preprocessData/data_preprocessing.py --format xml
```
This will generate the file 00_preprocessData/data/DATA.json which will be used for training. This file should be the same for all the following commands (even generation and evaluation).
WjazzDB requires specific pre-processing from the [csv files](http://mir.audiolabs.uni-erlangen.de/jazztube/downloads) provided by its authors. To train with WjazzDB run:
```
$ python3 A_preprocessData/wjazzDB_csv_to_xml.py
$ python3 A_preprocessData/data_preprocessing.py --format xml
```

## 2. Train MINGUS
To train MINGUS from scratch run:
```
$ python3 B_train/train.py 
```
The saved model will be in B_train/models. It is possible to visualize training details with tensorboard by running:
```
$ tensorboard --logdir=B_train/runs
```
The results are showed in your browser at http://localhost:6006/.

## 3. Generate music
To generate music with MINGUS you can run the following command. Please ensure that the arguments correspond to an already trained model. You can use the files in C_generate/xml4gen as a starting point. These files are also used for evaluation.
```
$ python3 C_generate/generate.py --xmlSTANDARD <path_to_xml_standard>
```
Output midi files will be in C_generate/generated_tunes.

To generate with our pre-trained model run:
```
$ python3 C_generate/generate.py --EPOCHS 100 --xmlSTANDARD <path_to_xml_standard>
```

## 4. Evaluate results
The evaluation metrics are computed by comparing the melodies generated by MINGUS and a corpus of generated samples, so to evaluate the model you should first choose some reference xml files and put them in D_eval/reference (you can randomly select some files from your original dataset). Then you should generate a corpus of melodies with MINGUS and evaluate the results:
```
$ python3 C_generate/generate.py --GENERATE_CORPUS
$ python3 D_evaluate/evaluate.py 
```
The evaluation metrics are available in D_evaluate/metrics.

To evaluate our pre-trained model run:
```
$ python3 C_generate/generate.py --GENERATE_CORPUS --EPOCHS 100
$ python3 D_evaluate/evaluate.py --EPOCHS 100
```
