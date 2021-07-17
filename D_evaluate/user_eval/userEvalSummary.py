#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:22:35 2021

@author: vincenzomadaghiele
"""
import json
import numpy as np
import seaborn as sns

# MODIFY

if __name__ == '__main__':
    
    userEval_path = open('metrics/00_userEvaluation/TUNES_STATS.json',)
    userEval = json.load(userEval_path)
        
    MINGUSlover = []
    MINGUSstudent = []
    MINGUSprof = []
    BebopLover = []
    BebopStudent =[]
    BebopProf = []
    OriginalLover = []
    OriginalStudent = []
    OriginalProf = []
    
    countLovers = 0
    countStudents = 0
    countProfessionals = 0
        
    for tune in userEval:
        if tune['origin'] == 'original':
            for rate in tune['ratings']:
                if rate['experience'] == 'Music lover':
                    OriginalLover.append(rate['rate'])
                if rate['experience'] == 'Music student':
                    OriginalStudent.append(rate['rate'])
                if rate['experience'] == 'Professional musician':
                    OriginalProf.append(rate['rate'])
        if tune['origin'] == 'MINGUS':
            for rate in tune['ratings']:
                if rate['experience'] == 'Music lover':
                    MINGUSlover.append(rate['rate'])
                if rate['experience'] == 'Music student':
                    MINGUSstudent.append(rate['rate'])
                if rate['experience'] == 'Professional musician':
                    MINGUSprof.append(rate['rate'])
        if tune['origin'] == 'BebopNet':
            for rate in tune['ratings']:
                if rate['experience'] == 'Music lover':
                    BebopLover.append(rate['rate'])
                if rate['experience'] == 'Music student':
                    BebopStudent.append(rate['rate'])
                if rate['experience'] == 'Professional musician':
                    BebopProf.append(rate['rate'])

    for rate in tune['ratings']:
        if rate['experience'] == 'Music lover':
            countLovers += 1
        if rate['experience'] == 'Music student':
            countStudents += 1
        if rate['experience'] == 'Professional musician':
            countProfessionals += 1


    userEvalSummary = {}
    userEvalSummary['original'] = {}
    userEvalSummary['MINGUS'] = {}
    userEvalSummary['BebopNet'] = {}
    userEvalSummary['original']['Music lover'] = np.mean(OriginalLover)
    userEvalSummary['original']['Music student'] = np.mean(OriginalStudent)
    userEvalSummary['original']['Professional musician'] = np.mean(OriginalProf)
    userEvalSummary['MINGUS']['Music lover'] = np.mean(MINGUSlover)
    userEvalSummary['MINGUS']['Music student'] = np.mean(MINGUSstudent)
    userEvalSummary['MINGUS']['Professional musician'] = np.mean(MINGUSprof)
    userEvalSummary['BebopNet']['Music lover'] = np.mean(BebopLover)
    userEvalSummary['BebopNet']['Music student'] = np.mean(BebopStudent)
    userEvalSummary['BebopNet']['Professional musician'] = np.mean(BebopProf)
    
    # Convert metrics dict to JSON and SAVE IT
    with open('metrics/00_userEvaluation/userEval_summary.json', 'w') as fp:
        json.dump(userEvalSummary, fp, indent=4)
        
    data = {
        'experience': [],
        'model': [],
        'mean': []
        }
    
    #sns.set_theme(style="whitegrid")
    #ax = sns.barplot(x="day", y="mean_rating", data=tips)

    import matplotlib.pyplot as plt

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # set height of bar
    musicLover = [np.mean(OriginalLover), np.mean(MINGUSlover), np.mean(BebopLover)]
    musicLoverStd = [np.std(OriginalLover), np.std(MINGUSlover), np.std(BebopLover)]
    student = [np.mean(OriginalStudent), np.mean(MINGUSstudent), np.mean(BebopStudent)]
    studentStd = [np.std(OriginalStudent), np.std(MINGUSstudent), np.std(BebopStudent)]
    professional = [np.mean(OriginalProf), np.mean(MINGUSprof), np.mean(BebopProf)]
    professionalStd = [np.std(OriginalProf), np.std(MINGUSprof), np.std(BebopProf)]

    # Set position of bar on X axis
    br1 = np.arange(len(musicLover))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    plt.style.use('seaborn')
    plt.grid(axis = 'x')
    #plt.rcParams['axes.axisbelow'] = True

    # Make the plot
    plt.bar(br1, musicLover, #color ='b', 
            width = barWidth,
            #edgecolor ='w', 
            label ='Music lover', yerr = musicLoverStd, capsize=.2)
    plt.bar(br2, student, #color ='g', 
            width = barWidth,
            #edgecolor ='w', 
            label ='Music student', yerr = studentStd, capsize=.2)
    plt.bar(br3, professional, #color ='r', 
            width = barWidth,
            #edgecolor ='w', 
            label ='Professional musician', yerr = professionalStd, capsize=.2)
    
    # Adding Xticks
    #plt.xlabel('Model', fontweight ='bold', fontsize = 20)
    plt.ylabel('Average rating', fontweight ='bold', fontsize = 20)
    plt.xticks([r + barWidth for r in range(len(musicLover))],
            ['Original', 'MINGUS', 'BebopNet'])

    plt.xticks(fontsize=20, fontweight ='bold')
    plt.yticks(fontsize=20)
    
    plt.legend(fontsize = 20)
    #plt.title("User evaluation of musical generation", fontsize=25, pad='2.0')
    plt.show()
