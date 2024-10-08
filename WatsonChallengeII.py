#!/usr/bin python3

'''
Created on Aug 6, 2024
@author: mayijie
'''

import os
import pandas as pd

class WatsonChallengeII:

    # show number of failed cases
    showCnt = 10

    csvFiles = ["/Users/mayijie/YijieDoc/DPM Docu/Watsonx Challenge/RQM/csv/2665.csv",
                "/Users/mayijie/YijieDoc/DPM Docu/Watsonx Challenge/RQM/csv/2597.csv",
                "/Users/mayijie/YijieDoc/DPM Docu/Watsonx Challenge/RQM/csv/2421.csv",
                "/Users/mayijie/YijieDoc/DPM Docu/Watsonx Challenge/RQM/csv/2389.csv",
                "/Users/mayijie/YijieDoc/DPM Docu/Watsonx Challenge/RQM/csv/2173.csv",
                "/Users/mayijie/YijieDoc/DPM Docu/Watsonx Challenge/RQM/csv/2167.csv",
                "/Users/mayijie/YijieDoc/DPM Docu/Watsonx Challenge/RQM/csv/1891.csv",
                "/Users/mayijie/YijieDoc/DPM Docu/Watsonx Challenge/RQM/csv/1583.csv",
                "/Users/mayijie/YijieDoc/DPM Docu/Watsonx Challenge/RQM/csv/1569.csv"]

    # combine all the dataframes
    df_list = []
    for csvFile in csvFiles:
        df = pd.read_csv(csvFile)
        # add the 'test_plan' column in the dataframe
        df['test_plan'] = os.path.basename(csvFile).split(".")[0]
        df_list.append(df)
    df = pd.concat(df_list)

    # dataframe self check
    if df['ID'].isnull().any() or not pd.api.types.is_integer_dtype(df['ID']):
        print ("Exist ID value is null or not integer type.")
        exit(0)
    if df['pass_points'].isnull().any() or not pd.api.types.is_integer_dtype(df['pass_points']):
        print ("Exist pass_points volue is null or not integer type.")
        exit(0)
    if df['total_points'].isnull().any() or not pd.api.types.is_integer_dtype(df['total_points']):
        print ("Exist total_points volue is null or not integer type.")
        exit(0)
    if (df['pass_points'] > df['total_points']).any():
        print ("Exist pass_points greater than total_points, need double check.")
        exit(0)

    # Get the failed test cases
    failed_records = df[df['defect'].notnull()]

    print ("There are " + str(df.shape[0]) + " test cases in " + str(len(csvFiles)) + " csv files, " + str(failed_records.shape[0]) + " cases are verified failed, here show the top " + str(showCnt) + " by failed_weight point:")

    # Add a column represent the 'failed_weight'
    failed_records = failed_records.copy()
    failed_records['failed_weight'] = failed_records['total_points'] - failed_records['pass_points']

    # sort the failed_records dataframe by descending 'failed_weight'
    failed_records = failed_records.sort_values(by='failed_weight', ascending=False)

    # get the top records and remove some useless columns
    top_records = failed_records.head(showCnt)
    top_records = top_records.drop('defect', axis=1)
    top_records = top_records.drop('pass_points', axis=1)
    top_records = top_records.drop('total_points', axis=1)

    print (top_records)



