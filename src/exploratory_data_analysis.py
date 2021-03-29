import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import string

DATA_DIR = '/Users/kkhare/Documents/Personal/MCS/DeepLearning/Project/iqvia_data/'
ENROLL_FILE = DATA_DIR + 'enroll_synth.dat'
CLAIMS_2019 = DATA_DIR + 'claims_2019.dat'
CLAIMS_2018 = DATA_DIR + 'claims_2018.dat'
CLAIMS_2017 = DATA_DIR + 'claims_2017.dat'
CLAIMS_2016 = DATA_DIR + 'claims_2016.dat'
CLAIMS_2015 = DATA_DIR + 'claims_2015.dat'
df_enroll = pd.read_csv(ENROLL_FILE, sep='|', low_memory=False)
df_claims2019 = pd.read_csv(CLAIMS_2019, sep='|', low_memory=False)
df_claims2018 = pd.read_csv(CLAIMS_2018, sep='|', low_memory=False)
df_claims2017 = pd.read_csv(CLAIMS_2017, sep='|', low_memory=False)
df_claims2016 = pd.read_csv(CLAIMS_2016, sep='|', low_memory=False)
df_claims2015 = pd.read_csv(CLAIMS_2015, sep='|', low_memory=False)
#print(df_enroll.head(5))


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