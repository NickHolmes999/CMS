#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: deanchanter
"""

import pandas as pd
import datetime
import numpy as np
import qualtricsapi as qa

#GLOBALS
token = qa.get_secrets()
sId_c = qa.SetSurveyId.survey_choices['Competency']
sId_a = qa.SetSurveyId.survey_choices['Awareness']

first_day_current_month= datetime.date.today().replace(day=1)

season_dict = [
                  '2021',
                  'Winter 2022',
                  'Spring 2022',
                  'Summer 2022',
                  'Fall 2022',
                  'Winter 2023',
                  'Spring 2023']

def reg_by_track(df_): 
    df_track = (df_.drop([0,1])
                .query("Dropped != 'Yes'")
                .reindex(columns=['Group', 'Season', "Repeat Learner Embedded",'Cohort', 'Halo', 'EndDate', 'Pillar_Complete', 'Email Embedded'])
                .groupby(['Cohort','Season']).count()
                .reset_index()
                .pivot('Cohort','Season')
                .loc[:,"Email Embedded"]
                .drop(columns = "WL")
                .reindex(season_dict, axis=1)
                .dropna(axis=1, how='all')
            )
          
    return df_track
 

def main():

    df_aw_raw = qa.getResults(sId_a,token)
    #df_aev_raw = qa.getResults(sId_aev,token)
    df_comp_raw = qa.getResults(sId_c,token)

    df_aw_raw.to_csv("aw_dump_raw.csv")
    df_comp_raw.to_csv("comp_dump_raw.csv")
    #df_aev_raw.to_csv("aev_dump_raw.csv")


    #df_comp_raw = pd.read_csv("comp_dump_raw.csv")
    #df_aw_raw = pd.read_csv("aw_dump_raw.csv")
    #df_aev = pd.read_csv("aev_dump_raw.csv")


    today = datetime.date.today()

    df_awareness_by_track = reg_by_track(df_aw_raw)
    df_competency_by_track = reg_by_track(df_comp_raw).drop(index = 'CMS 101 Procurement Spend Optimization (PSO)')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df_awareness_by_track)
    print(df_competency_by_track)
    
    pd.reset_option('all', silent=True)

    with pd.ExcelWriter(f"reg_report_{today}.xlsx") as writer:
        df_awareness_by_track.to_excel(writer, sheet_name="Awareness Registration")
        df_competency_by_track.to_excel(writer, sheet_name="Competency Registration")


if __name__ == '__main__':
    main()