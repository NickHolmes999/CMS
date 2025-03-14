#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:23:55 2023

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
sId_aev = 'SV_aVEN5GAHH4W8rTU'
first_day_current_month= datetime.date.today().replace(day=1)

current_seasons = ['Winter 2023','Spring 2023']
season = 'Spring 2023'

def clean(df_, pillar):
    
    def pillar_complete(df_):
        if (pillar == "Awareness"):
            return np.where((df_["Verified Complete Embedded"]=="Yes") & (df_["Halo"]=="Yes"),"Yes","No")
        else:
            return np.where(df_['Verified Complete']=='Yes','Yes','No')
    
    
    def verify(df_):
        if (pillar == "Awareness"):
           df_["Verified Complete"] = df_["Verified Complete Embedded"].fillna("No")
        else:
            df_["Verified Complete"] = df_["Verified Complete"].fillna("No")
        return df_

    return (df_.drop([0,1])
                .query("Dropped != 'Yes'")
                #.query('Chort' )
                 .query("Group == ['AMG','BOG', 'EADG','ESSG','ICPG','ISPG','IUSG','OIT Front Office']")
                 # .drop_duplicates(subset="Email Embedded")
                  .assign(EndDate = lambda df_: pd.to_datetime(df_['EndDate'], infer_datetime_format=True))
                  .assign(Pillar_Complete = lambda df_ : pillar_complete(df_)) 
                  #.apply((lambda df_: pd.to_datetime(df_['EndDate'], infer_datetime_format=True)), axis =1, result_type = 'broadcast')
                  .pipe(verify)
                  .fillna({'Repeat Learner Embedded':'Drop_Me'})
                  .reindex(columns=['Group', 'Season', "Repeat Learner Embedded",'Cohort', 'Halo', 'EndDate', 'Awareness', 'Pillar_Complete', 'Verified Complete', 'Email Embedded'])
                  )

def complete(df_, pillar):
        return (df_.query("Pillar_Complete == 'Yes'")
                    .query("`Repeat Learner Embedded` != 'TRUE'")
                    .groupby("Group").count()
                    .loc[:,"Pillar_Complete"]
                    .to_frame()
                    .rename(columns={"Pillar_Complete": f"{pillar} Pillar Completion"})
                    
                ) 

def this_season(df_, season):
        
        df_season = ( df_
                    .query('Season == @season')
                    .groupby(["Group","Repeat Learner Embedded","Verified Complete"]).count()
                    .reset_index()
                    .pivot("Group",["Verified Complete","Repeat Learner Embedded"])
                    .loc[:,"Season"]
                    .rename(columns={"FALSE":"New Learners","TRUE":"Repeat Learners", "Drop_Me": ""})
                    .rename(columns={"No":f"{season} Currently Enrolled","Yes":f"{season} Completed"})
                )
        df_season.columns = ['-'.join(str(s).strip() for s in col if s) for col in df_season.columns]
        
        return df_season 

def total(df_):
    df_.loc["Total",:] = df_.sum()
    df_=df_.astype('int', errors='ignore')
    return df_
    

def montly_report(df_, pillar):

    df_clean = df_.pipe(clean, pillar)
    
    df_pillar_complete = df_clean.pipe(complete, pillar=pillar)

    df_this_season = pd.DataFrame()
    for s in current_seasons: 
        if (s in df_clean['Season'].values):
            df_s = df_clean.pipe(this_season, season = s)
            df_this_season = pd.concat([df_this_season, df_s], axis=1)
            '''
            df_last_month = (df_clean.query("EndDate < @first_day_current_month")
                        .pipe(this_season)
                        )
                    
            df_change = ((df_this_season - df_last_month)
                    .add_suffix(" change since last month")
                    )
            '''

    df_pillar_complete = (df_pillar_complete
                .join(df_this_season.filter(regex=(".*Completed.*")))
                .fillna(0)
                .pipe(total)
    )

    df_pillar_enrolled = (df_this_season.filter(regex=(".*Enrolled.*"))
                            .fillna(0)
                            .pipe(total)
    )
   
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df_pillar_complete)
    print(df_pillar_enrolled)
    
    pd.reset_option('all', silent=True)
 
    return df_pillar_complete, df_pillar_enrolled





def montly_okrs(df_awareness, df_competency):
    season = 'Spring 2023'

    new_awareness_learners_this_season = (df_awareness
                              .pipe(clean, pillar='Awareness')
                            .query("Season == @season")
                           .query("`Repeat Learner Embedded` != 'TRUE'")
                            .loc[:,"Season"]
                            .count()
                            )



    awareness_learners = (df_awareness
                      .pipe(clean,pillar='Awareness')
                           .query('Pillar_Complete == "Yes"')
                             .groupby('Repeat Learner Embedded').count()
                             .loc[:,'Pillar_Complete'].T
                             .rename(index={"FALSE":"New Learners","TRUE":"Repeat Learners"})
                             )
                         

    repeat_awareness_learners_percentage = ((awareness_learners[1] /  awareness_learners[0])*100).astype('int')





    competency_learners=(df_competency
                       .pipe(clean,pillar='competency')
                       .dropna(subset = ['Season'])
                       .query('Season != "WL"')
                       .query('Cohort != "CMS 101 Procurement Spend Optimization (PSO)"')
                       .loc[:,'Email Embedded']
                             .count()
                             )


    awareness_learners_2022 = 223
                             
    competency_convert_percentage = (competency_learners/ awareness_learners_2022)*100

    okr_data = [new_awareness_learners_this_season,
            repeat_awareness_learners_percentage,
            competency_convert_percentage]

    okr_index = [f"new awareness {season} learners",
            "repeat awareness learners percentage",
            "competency convertion percentage"]

    df_okrs = pd.DataFrame(data=okr_data, index= okr_index, columns=['OKRs']).round(0).astype('int')

    print(df_okrs)
    return df_okrs




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

    df_monthly_report_awareness_complete, df_monthly_report_awareness_enrolled = montly_report(df_aw_raw, 'Awareness')

    df_monthly_report_competency_complete, df_monthly_report_competency_enrolled = montly_report(df_comp_raw, 'Competency')

    df_okrs = montly_okrs(df_aw_raw, df_comp_raw)

    today = datetime.date.today()

    with pd.ExcelWriter(f"month_report_{today}.xlsx") as writer:
        df_monthly_report_awareness_complete.to_excel(writer, sheet_name="Awareness Complete")
        df_monthly_report_awareness_enrolled.to_excel(writer, sheet_name="Awareness Enrolled")
        df_monthly_report_competency_complete.to_excel(writer,sheet_name="Competency Complete")
        df_monthly_report_competency_enrolled.to_excel(writer,sheet_name="Competency Enrolled")
        df_okrs.to_excel(writer,sheet_name='OKRs')

if __name__ == '__main__':
    main()