import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

import string

## Load Enrollment and Claims Data
DATA_DIR = '/Users/kkhare/Documents/Personal/MCS/DeepLearning/Project/iqvia_data/'
ENROLL_FILE = DATA_DIR + 'enroll_synth.dat.csv'
CLAIMS_2019 = DATA_DIR + 'claims_2019.dat'
CLAIMS_2018 = DATA_DIR + 'claims_2018.dat'
CLAIMS_2017 = DATA_DIR + 'claims_2017.dat'
CLAIMS_2016 = DATA_DIR + 'claims_2016.dat'
CLAIMS_2015 = DATA_DIR + 'claims_2015.dat'
df_enroll = pd.read_csv(ENROLL_FILE, sep=',', low_memory=False)
df_claims2019 = pd.read_csv(CLAIMS_2019, sep='|', low_memory=False)
df_claims2018 = pd.read_csv(CLAIMS_2018, sep='|', low_memory=False)
df_claims2017 = pd.read_csv(CLAIMS_2017, sep='|', low_memory=False)
df_claims2016 = pd.read_csv(CLAIMS_2016, sep='|', low_memory=False)
df_claims2015 = pd.read_csv(CLAIMS_2015, sep='|', low_memory=False)
#print(df_enroll.head(5))

## Add year and create a single dataset for claims
df_claims2015["year"] = 2015
df_claims2016["year"] = 2016
df_claims2017["year"] = 2017
df_claims2018["year"] = 2018
df_claims2019["year"] = 2019

list_of_claims = [df_claims2015, df_claims2016, df_claims2017, df_claims2018, df_claims2019]

df_claims = pd.concat(list_of_claims)
print("Shape of Claims{}".format(df_claims.shape))


# Distribution of patients across regions
rd = df_enroll["pat_region"].value_counts().plot(kind="pie", autopct="%1.1f%%")
rd.set_title("Distribution of patients across regions")

# Distribution of patients gender
rd = df_enroll["der_sex"].value_counts().plot(kind="pie", autopct="%1.1f%%")
rd.set_title("Distribution of patients' gender")

# Distribution of Age
df_enroll["age"] = 2021 - df_enroll["der_yob"]

rd = df_enroll[df_enroll["der_yob"] > 1900]["age"].plot(kind='hist', bins=15)
rd.set_title("Distribution of patients' age")

# Get the count of claims paid (and denied)
df_claims["pmt_st_cd"].value_counts()

# number of diagnosis populated in each claim
diag_cols = ["diag1", "diag2", "diag3", "diag4", "diag5", "diag6", "diag7", "diag8", "diag9", "diag10", "diag11", "diag12"]
df_claims["num_of_diag"] = df_claims[diag_cols].notnull().sum(axis=1)
df_claims["num_of_diag"].mean()

# number of icdprc populated in each claim
icdprc_cols=["icdprc1", "icdprc2", "icdprc3", "icdprc4", "icdprc5", "icdprc6", "icdprc7", "icdprc8", "icdprc9", "icdprc10", "icdprc11", "icdprc12"]
df_claims["num_of_icdprc"] = df_claims[icdprc_cols].notnull().sum(axis=1)
df_claims["num_of_icdprc"].mean()

diag = []
for colname in diag_cols:
    diag.extend(pd.unique(df_claims[colname]))
print(len(np.unique(diag)))
# 22138

prc = []
for colname in icdprc_cols:
    prc.extend(pd.unique(df_claims[colname]))
print(len(np.unique(prc)))
# 926


# number of claims with same day service
sum(df_claims["from_dt"] == df_claims["to_dt"])
# 2378556 out of 2438054 i.e. 97.5%

# Distribution of charges
rd = df_claims["charge"].plot(kind='hist', bins=15)
rd.set_title("Distribution charges")

# Log charges makes more sense
# filtering out rows where charges are less than 1
rd = np.log10(df_claims[df_claims["charge"] > 1]["charge"]).plot(kind='hist', bins=15)
rd.set_title("Distribution of Log of Charges")

# Distribution of Paid amounts
# filtering out rows where paid are less than 1
rd = np.log10(df_claims[df_claims["paid"] > 1]["paid"]).plot(kind='hist', bins=25)
rd.set_title("Distribution of Log of Paid")
plt.show()


# Checking the unique number of patients in the datasets
print(len(pd.unique(df_enroll['pat_id'])))
# 30000
print(len(pd.unique(df_claims2015['pat_id'])))
# 18927
print(len(pd.unique(df_claims2016['pat_id'])))
#21483
print(len(pd.unique(df_claims2017['pat_id'])))
#15190
print(len(pd.unique(df_claims2018['pat_id'])))
#6445
print(len(pd.unique(df_claims2019['pat_id'])))
#4884

#################################
## Create dataset for modeling ##
#################################
# each patient with a single record
# x -> aggregation of rows from all except last qtr
# y -> charged in last qtr

# 1. first assign quarter to each record
df_claims["quarter"] = pd.PeriodIndex(pd.to_datetime(df_claims["to_dt"]), freq = 'Q')

# 2. find the latest quarter for each patient
df_patient_last_quarter = df_claims.groupby('pat_id')["quarter"].max().reset_index()

# 3. Find claims for patient in last quarter and not in last quarter

df_claims_last_quarter = df_claims[df_claims.set_index(['pat_id','quarter']).index.isin(df_patient_last_quarter.set_index(['pat_id','quarter']).index)]
df_claims_not_last_quarter = df_claims[~df_claims.set_index(['pat_id','quarter']).index.isin(df_patient_last_quarter.set_index(['pat_id','quarter']).index)]

# 4. confirm many unique patients are in both sets
len(pd.unique(df_claims_last_quarter['pat_id']))
# 27226
sum(np.isin(pd.unique(df_claims_not_last_quarter["pat_id"]), pd.unique(df_claims_last_quarter["pat_id"])))
# 24970

# Create y variable for each patient
df_y = df_claims_last_quarter.groupby('pat_id')["charge"].sum().reset_index()

# Create x variable for each patient??
