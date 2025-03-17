#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:58:02 2023

@author: deanchanter
"""


import requests
import csv
import datetime
import json
import pandas as pd
import numpy as np
import zipfile
import io
import argparse
import configparser
import os


#Setting Global Variable for DataCenter
DC='cms.gov1'

def get_secrets(cfg_file='qualtrics_key.ini'):
    config=configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__),cfg_file))
    try:
        return config['API']['token']
    except:
        raise SystemExit("could not find qualtrics key")
        

def getResults(surveyId, token):
    fileFormat = 'csv'
    requestCheckProgress = 0
    progressStatus = "in progress"
    useLabels = True
    baseUrl = "https://{0}.qualtrics.com/API/v3/responseexports/".format(DC)
    headers = {
        "content-type": "application/json",
        "x-api-token": token,
    }

    # Step 1: Creating Data Export
    downloadRequestUrl = baseUrl
    downloadRequestPayload = json.dumps({"format": fileFormat, "surveyId": surveyId, "useLabels": useLabels})
    downloadRequestResponse = requests.request("POST", downloadRequestUrl, 
                                data=downloadRequestPayload, headers=headers)
    
    print("API Response Status Code:", downloadRequestResponse.status_code)
    print("API Response Content:", downloadRequestResponse.text)

    if downloadRequestResponse.status_code != 200:
        raise Exception(f"API request failed with status code {downloadRequestResponse.status_code}")

    response_json = downloadRequestResponse.json()
    
    if "result" not in response_json:
        print("Full API response:", json.dumps(response_json, indent=2))
        raise KeyError("'result' key not found in API response")
    
    if "id" not in response_json["result"]:
        print("Full API response:", json.dumps(response_json, indent=2))
        raise KeyError("'id' key not found in API response 'result'")

    progressId = response_json["result"]["id"]
    print(f"Progress ID: {progressId}")

    # Step 2: Checking on Data Export Progress and waiting until export is ready
    while requestCheckProgress < 100 and progressStatus != "complete":
        requestCheckUrl = baseUrl + progressId
        requestCheckResponse = requests.request("GET", requestCheckUrl, headers=headers)
        requestCheckProgress = requestCheckResponse.json()["result"]["percentComplete"]
        print("Download is " + str(requestCheckProgress) + " complete")

    # Step 3: Downloading file
    requestDownloadUrl = baseUrl + progressId + '/file'
    requestDownload = requests.request("GET", requestDownloadUrl, headers=headers, stream=True)

    # Step 4: Reading the zip file and Creating a DF
    with zipfile.ZipFile(io.BytesIO(requestDownload.content), mode='r') as archive:
        #assuming only the csv is in the zip 
        filename = archive.namelist()[0]
        data = archive.read(filename)
        dfResults = pd.read_csv(io.BytesIO(data))
        
    #make sure responseId is in the right case
    if "ResponseId" not in dfResults.columns[dfResults.columns.str.startswith('ResponseI')]:
        cols = dfResults.columns[dfResults.columns.str.startswith('ResponseI')]
        if len(cols) > 0:
            col = cols[0]
            dfResults.rename(columns = {col: 'ResponseId'}, inplace=True)
    return dfResults

   
def pull_file_contents(filename):
    file_data = []
    with open(filename, 'rt') as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            file_data.append(row)
    return file_data


def write_log(filename, message):
    now = datetime.date.today().strftime('%Y-%m-%d-%H.%M.%S')
    with open(filename, 'a') as logfile:
        logfile.write(now + ': ' + message + '\n')


def make_update(responseId, token, surveyId, EDObj):
    url = "https://{0}.qualtrics.com/API/v3/responses/{1}".format(
        DC, responseId)
    payload = json.dumps({
        "surveyId": surveyId,
        "resetRecordedDate": False,
        "embeddedData": EDObj
    })
    print(f"Sending update request for ResponseId: {responseId}")
    print(f"Payload: {payload}")
    response = requests.put(url=url,
                            headers={
                                'X-API-TOKEN': token,
                                'Content-Type': 'application/json'
                            },
                            data=payload
                            )
    print(f"Status: {response.status_code}")
    print(f"Response Body: {response.text}")
    if response.status_code != 200:
        write_log("errors.txt", f"ResponseId: {responseId}, Status: {response.status_code}, Body: {response.text}")
        return responseId
    else:
        return 0




def check_update(responseId,token, surveyId,EDObj):
    return_error = 0
    url = "https://{0}.qualtrics.com/API/v3/surveys/{1}/responses/{2}".format(
        DC, surveyId, responseId)
    response = requests.get(url=url,
                            headers={
                                'X-API-TOKEN': token,
                                'Content-Type': 'application/json'
                            })
    print(response)
    for key in EDObj.keys():
        if EDObj[key] != response.json()["result"]["values"].get(key):
            write_log("errors.txt", "{0}:{1} update failed got {2} instead of {3}".format(
                responseId,key,response.json()["result"]["values"].get(key),EDObj[key]))
            print("{0}:{1} update failed".format(responseId, key))
            return_error = responseId  
    return return_error
    
def bulkUpdateFromCSV(csv,surveyId, token):

    bulkUpdate(pd.read_csv(csv),surveyId, token)
    
def bulkUpdate(df,surveyId, token):
    #need to fix for empty columns
    errList = []
    if df.index.name != 'ResponseId':
        df.set_index('ResponseId',inplace=True)
    updateDict = df.to_dict('index')
    totalUpdates = len(updateDict)
    count = 0
    for ResponseId in updateDict:
        count += 1
        data = {key:value for (key, value) in updateDict[ResponseId].items() if not pd.isna(value)}
        print("Updating row {0}/{1}, ResponseId: {2}".format(count,
                    totalUpdates, data))
        err = make_update(ResponseId,token, surveyId,data)
        if err:
            errList.append(err)
        err = check_update(ResponseId,token, surveyId,data)
        if err:
            errList.append(err)
    print(errList)

class SetSurveyId(argparse.Action):
    survey_choices = {'Awareness_Legacy' : 'SV_9EtYLCBZb9rJ8Ds', 
                      'Competency_Legacy': 'SV_cYBwZAxxKeBCYbI',
                      'Awareness' : 'SV_5vR6fdtnbcsJeku',
                      'Competency': 'SV_7ZCO91v8YtPMR5I',
                      'Business_Agility': 'SV_8entAzhlK6dLwua',
                      'HCD': 'SV_eUUr5FsYmS1rpkO'}
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.survey_choices[values])

def find_halo(surveyId, token, halo_list):
    df_halo = pd.read_csv(halo_list).loc[:,'Passed_Halo'].str.strip().str.lower()


    #df_halo_grandfather = pd.read_csv("HaloGrandFather.csv").loc[:,'Email Embedded'].str.lower()

    df_aw_halo = (getResults(surveyId, token)
                    .drop([0,1])
                    .loc[:,['ResponseId','Email Embedded','Halo']]
                    .assign(email_clean = lambda df_: df_['Email Embedded'].str.strip().str.lower())
                    .drop(columns='Email Embedded')
                    .rename(columns={'email_clean':'Email Embedded'})

    )
    df_aw_halo.to_csv('halo_check.csv')
    df_aw_halo['Halo_check'] = df_aw_halo['Email Embedded'].isin(df_halo.to_numpy())

    return (df_aw_halo
                .replace({True : 'Yes', False : 'No'})
                .query("Halo != Halo_check")
                .set_index('ResponseId')
                .drop(columns= ['Halo', 'Email Embedded'])
                .rename(columns={'Halo_check':'Halo'})
                    )



import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def find_repeat_learners(surveyId, token):
    season_dict = pd.api.types.CategoricalDtype([
        '2021',
        'Winter 2022',
        'Spring 2022',
        'Summer 2022',
        'Fall 2022',
        'Winter 2023',
        'Spring 2023',
        'Summer 2023',
        'Fall 2023',
        'Winter 2024',
        'Spring 2024',
        'Summer 2024',
        'Fall 2024',
        'Winter 2025',
        'Spring 2025',
        'Summer 2025',
    ], ordered=True)

    def needs_update_map(df_):
        logger.debug(f"Processing {len(df_)} records")
        predicted_repeat_learner = pd.Series('FALSE', index=df_.index)


        for email, group in df_.groupby('Email Embedded'):
            logger.debug(f"Processing group for email: {email}")
            verified_complete_index = group[group['Verified Complete'] == 'Yes'].index
            if not verified_complete_index.empty:
                first_verified_complete = verified_complete_index.min()
                logger.debug(f"First 'Verified Complete' for {email}: {first_verified_complete}")
                predicted_repeat_learner.loc[group.index[group.index > first_verified_complete]] = 'TRUE'
                logger.debug(f"Set {sum(group.index > first_verified_complete)} records as repeat learners for {email}")

        updates_needed = (predicted_repeat_learner != df_['Repeat Learner'])
        logger.debug(f"Total updates needed: {sum(updates_needed)}")

        return np.where(updates_needed, predicted_repeat_learner, np.nan)

    initial_df = getResults(surveyId, token)
    logger.debug(f"Initial number of records: {len(initial_df)}")
    
    df = initial_df.drop([0,1])
    logger.debug(f"After dropping first two rows: {len(df)}")
    
    df = df.assign(Season = lambda df_ : df_['Season'].astype(season_dict))
    logger.debug(f"After assigning Season: {len(df)}")
    
    df = df.sort_values(by=['Email Embedded', 'Season'], kind='stable')
    logger.debug(f"After sorting: {len(df)}")
    
    df = df.dropna(subset=['Season'])
    logger.debug(f"After dropping rows with NaN Season: {len(df)}")
    
    logger.debug(f"Columns in dataframe: {df.columns.tolist()}")
    
    df = df.assign(needs_update = lambda df_ : needs_update_map(df_))
    
    result = (df.loc[:,['ResponseId', 'needs_update']]
              .rename(columns={'needs_update': 'Repeat Learner'})
              .set_index('ResponseId')
              .dropna())
    
    logger.debug(f"Final number of records needing update: {len(result)}")
    
    return result





def update_reapeat_learners(surveyId, token):
    df = find_repeat_learners(surveyId, token)
    bulkUpdate(df, surveyId, token)



def update_halo(surveyId, token, halo_list):
    df = find_halo(surveyId, token, halo_list)
    bulkUpdate(df, surveyId, token) 

def main():
    survey_choices = SetSurveyId.survey_choices.keys()
    parser = argparse.ArgumentParser(description="Interact with the Qualtrics API")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--bulkUpdateFromCSV', help="update from a csv file", type=str)
    action.add_argument('--reapeat_learners', help="search for repeat learners and make the updates", 
                        action='store_true')
    action.add_argument('--Halo', help="search for mismatched halo and make the updates", type=str)
    action.add_argument('--update_all', help='Update both Repeat Learners and Halo', type=str)
    parser.add_argument('--ini', help='location of the ini file', type=str)
    s_id = parser.add_mutually_exclusive_group(required=True)
    s_id.add_argument('--surveyId', help='Choices {0}'.format(survey_choices),
                      choices=survey_choices, action=SetSurveyId, dest='surveyId')
    s_id.add_argument('--manuel_surveyId', help='Manual surveyId Cannot be used with --survey', 
                        type=str, dest='surveyId')
    args = parser.parse_args()
    
    if args.ini is not None:
        token = get_secrets(cfg_file=args.ini)
    else:
        token = get_secrets()

    if args.bulkUpdateFromCSV is not None:
        bulkUpdateFromCSV(args.bulkUpdateFromCSV, args.surveyId, token)
    if args.reapeat_learners: 
        update_reapeat_learners(args.surveyId, token)
    if args.Halo is not None:
        update_halo(args.surveyId, token, args.Halo)
    if args.update_all is not None:
        update_reapeat_learners(args.surveyId, token)
        update_halo(args.surveyId, token, args.update_all)

if __name__ == '__main__':
    main()