import qualtricsapi as qa
import pandas as pd
import numpy as np
import re

token = qa.get_secrets()
#sId_c = qa.SetSurveyId.survey_choices['Competency']
sId_a = qa.SetSurveyId.survey_choices['Awareness']
sId_ev = 'SV_aVEN5GAHH4W8rTU'
'''
df_aw = qa.getResults(sId_a,token)
df_ev = qa.getResults(sId_ev,token)
df_aw.to_csv("aw_dump219.csv")
df_ev.to_csv("ev_219.csv")
'''
#df_comp = qa.getResults(sId_c,token)
#df_comp.to_csv("comp_dump215.csv")

# print(df)
#df.to_csv("aw_dump.csv")

#df_comp = pd.read_csv("comp_dump215.csv")
df_aw = pd.read_csv("aw_dump219.csv")
df_ev = pd.read_csv("ev_219.csv")

aw_cols = ['ResponseId',
            'Group', 
            'Season', 
            'Verified Complete Embedded', 
            'Repeat Learner Embedded',
            'Cohort', 
            'Halo', 
            'EndDate', 
            'Email Embedded', 
            'Dropped',
            'Motivation_1',
	        'Motivation_2',
    	    'Motivation_3',
            'Motivation_4',
            'Motivation_5',
            'Motivation_6',
            'Motivation_7',
            'Motivation_8',
            'Motivation_9',
            'Motivation_9_TEXT',
            'Topic Interest',
            'Prior Participation',
            'Prior Track(s)_14',
            'Prior Track(s)_7',
            'Prior Track(s)_12',
            'Prior Track(s)_8',
            'Prior Track(s)_9',
            'Prior Track(s)_10',
            'Prior Track(s)_13',
            'Prior Track(s)_11']



ev_cols = [ 
       'EndDate',
       'Email', 
       'Group', 
       'Learning Track', 
        'A-E-CMS Cohort',  
        'ROTI',
       'Increasing ROTI', 
       'NPS', 'Course Content_4', 
       'Course Content_1',
       'Course Content_2', 
       'Course Content_3', 
       'Course Content_5',
       'Course Content_9', 
       'Liked Most', 
       'Liked Least', 
       'How to Improve',
       'Competency Interest', 
       'COP Interest', 
       'Certificate Interest']


df_ev_clean = (df_ev.query('`Learning Track` == "CMS 101"')
                    .loc[:,ev_cols]
                    .set_index("Email")
)

df_aw['A-R'] = df_aw['A R-Cloud'].fillna(df_aw['A R-CMS 101']).fillna(df_aw['A R-Cybersecurity']).fillna(
        df_aw['A R-Data Science']).fillna(df_aw['A R-HCD']).fillna(df_aw['A R-Leadership']).fillna(df_aw['A R-Product Mgmt']).fillna(df_aw['A Cohort Season'])
    
df_aw['A-R'] = df_aw['A-R'].replace('Please notify.*',"WL", regex=True)
df_aw['A-R'] = df_aw['A-R'].replace('Please add.*',"WL",regex=True)
df_aw['A-R'] = df_aw['A-R'].replace('2021','2021')
df_aw['A-R'] = df_aw['A-R'].replace('Winter 2022*','Winter 2022', regex=True)
df_aw['A-R'] = df_aw['A-R'].replace('Spring 2022*','Spring 2022', regex=True)
df_aw['A-R'] = df_aw['A-R'].replace('Summer 2022*','Summer 2022', regex=True)
df_aw['A-R'] = df_aw['A-R'].replace('Fall 2022*','Fall 2022', regex=True)

#print(df_aw['A-R'].head(10))

df_cms101_clean = (df_aw
                    .assign( Season = lambda df_ : df_['Season'].fillna(df_['A-R']))
                    .query('Season != "WL"')
                    .query('Cohort == "CMS 101"')
                    .loc[:,aw_cols]
                    .set_index("Email Embedded")
                    .join(df_ev_clean, how = "outer", rsuffix="_registration", lsuffix='_evaluation')
                    .reset_index(drop=True)
)

#print(df_cms101_clean.head(10))

print(df_ev_clean.columns)

df_ev_rollup = (df_ev_clean
                .assign(ROTI = lambda x: re.sub(r'^(\d+)-.+',r'\1' ,str(x)))
                .assign(**{"Course Content_4" : lambda x: re.sub(r'^(\d+)-.+',r'\1' ,str(x))})
                .assign(**{"Course Content_1" : lambda x: re.sub(r'^(\d+)-.+',r'\1' ,str(x))})
                .assign(**{"Course Content_2" : lambda x: re.sub(r'^(\d+)-.+',r'\1' ,str(x))})
                .assign(**{"Course Content_3" : lambda x: re.sub(r'^(\d+)-.+',r'\1' ,str(x))})
                .assign(**{"Course Content_5" : lambda x: re.sub(r'^(\d+)-.+',r'\1' ,str(x))})
                .assign(**{"Course Content_9" : lambda x: re.sub(r'^(\d+)-.+',r'\1' ,str(x))})
                            )
	


print(df_ev_rollup)

#df_cms101_clean.to_csv("cms101_clean.csv")

#df_cms101_rollup = (df_cms101_clean[["NPS","ROTI"]].dropna().astype("int").mean())
#print(df_cms101_rollup)