#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:10:04 2022

@author: deanchanter
"""



import pandas as pd
import sys
import numpy as np

def findRepeatLearners(file):
    dfFRL = pd.read_csv(file)
    #drop first two junk rows and columns we dont need ToDo
    dfFRL = dfFRL.drop([0,1])
    #dfFRL = dfFRL.drop(columns=[''])
    
    
    #Create a Reg Helper Col for sorting ToDo: Clean this up
    
    dfFRL['SeasonOrg'] = dfFRL['Season']
    
    dfFRL['Season'] = dfFRL['Season'].replace('Please notify',float('inf'), regex=True)
    dfFRL['Season'] = dfFRL['Season'].replace(np.nan,float('inf'))
    dfFRL['Season'] = dfFRL['Season'].replace('2021',0)
    dfFRL['Season'] = dfFRL['Season'].replace('Winter 2022',1, regex=True)
    dfFRL['Season'] = dfFRL['Season'].replace('Spring 2022',2, regex=True)
    dfFRL['Season'] = dfFRL['Season'].replace('Summer 2022',3, regex=True)
    dfFRL['Season'] = dfFRL['Season'].replace('Fall 2022',4, regex=True)
    dfFRL['Season'] = dfFRL['Season'].replace('Winter 2023',5, regex=True)
    
   
    
    #Sort
    dfFRL = dfFRL.rename_axis('dfFRLIndex').sort_values(by=['Season','dfFRLIndex'])
    
    #search for repeat learners by email address
    #dfFRL['dup'] = dfFRL['Email Embedded'].duplicated('first')
    
    
    #Mark the lines the "should" be labled "Repeat Learner"
    dfFRL['emailDup'] = dfFRL['Email Embedded'].duplicated()
    dfFRL['Repeat Learner Embedded'] = dfFRL['Repeat Learner Embedded'].map({'TRUE':True,'FALSE':False})
    

    #Mark Lines that needs "fixing"
    #dfFRL['needsUpdate'] = ((dfFRL['emailDup']) & ~(dfFRL['Repeat Learner']) | (~(dfFRL['repeatLearner']) & (dfFRL['Group'] == 'Repeat Learner'))
    dfFRL['needsUpdate'] = ~((dfFRL['emailDup']) == (dfFRL['Repeat Learner Embedded']))
    
   
  
    
    #print(dfFRL[['emailDup','Repeat Learner Embedded','needsUpdate', 'ResponseId']].head(50))
    dfChanges = dfFRL[['ResponseId','emailDup']].loc[dfFRL['needsUpdate'] == True].rename(columns = {'emailDup': 'Repeat Learner Embedded'})
    print (dfChanges)
    return dfChanges

def main():
    file = "/Users/deanchanter/Downloads/Registration+Survey+ +WR+Awareness+Pillar_November+30,+2022_10.24.csv"
    needsFix = findRepeatLearners(file)
    #needsFix = findRepeatLearners(sys.argv[1])

    needsFix.to_csv('needsFix.csv')
    

main()