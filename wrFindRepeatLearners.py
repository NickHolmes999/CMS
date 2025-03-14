#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:10:04 2022

@author: deanchanter
"""



import pandas as pd
import sys

def findRepeatLearners(file):
    dfFRL = pd.read_csv(file)
    #drop first two junk rows and columns we dont need ToDo
    dfFRL = dfFRL.drop([0,1])
    #dfFRL = dfFRL.drop(columns=[''])
    
    
    #Create a Reg Helper Col for sorting ToDo: Clean this up
    dfFRL['A-R'] = dfFRL['A R-Cloud'].fillna(dfFRL['A R-CMS 101']).fillna(dfFRL['A R-Cybersecurity']).fillna(
        dfFRL['A R-Data Science']).fillna(dfFRL['A R-HCD']).fillna(dfFRL['A R-Leadership']).fillna(dfFRL['A R-Product Mgmt'])
    
    dfFRL['A-R'] = dfFRL['A-R'].replace('Please notify',float('inf'), regex=True)
    dfFRL['A-R'] = dfFRL['A-R'].replace('2021',0)
    dfFRL['A-R'] = dfFRL['A-R'].replace('Winter 2022*',1, regex=True)
    dfFRL['A-R'] = dfFRL['A-R'].replace('Spring 2022*',2, regex=True)
    dfFRL['A-R'] = dfFRL['A-R'].replace('Summer 2022*',3, regex=True)
    
    
    #Sort
    dfFRL = dfFRL.rename_axis('dfFRLIndex').sort_values(by=['A-R','dfFRLIndex'])
    
    #search for repeat learners by email address
    dfFRL['dup'] = dfFRL['Email'].duplicated(False)
    
    
    #Mark the lines the "should" be labled "Repeat Learner"
    dfFRL['emailDup'] = dfFRL['Email'].duplicated()
    
    
    #Mark Lines that needs "fixing"
    #dfFRL['needsUpdate'] = ((dfFRL['emailDup']) & ~(dfFRL['Repeat Learner']) | (~(dfFRL['repeatLearner']) & (dfFRL['Group'] == 'Repeat Learner'))
    dfFRL['needsUpdate'] = (dfFRL['emailDup']) ^ (dfFRL['Repeat Learner'])
    
    print(dfFRL[['dup','Repeat Learner','needsUpdate', 'ResponseId']].head(50))
    
    return dfFRL

def main():
    
    needsFix = findRepeatLearners(sys.argv[1])

    needsFix.to_csv('needsFix.csv')
    

main()