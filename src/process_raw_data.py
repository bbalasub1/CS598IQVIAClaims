'''
this code is run first to read raw data,
 and write into a single pickle file (that is read for subsequent operations)

read various claims data file 
filter to track only patients in the database between Jan 1, 2015 to June 30, 2015
save as compressed pickle
'''

import pandas, numpy
import urllib.request
import os

def read_raw_data_files(DATADIR):
    out = []
    for yr in [2015, 2016, 2017, 2018, 2019]:
        fname = f"{DATADIR}/claims_{yr}.dat"
        #print(f"reading {fname}")
        claims = pandas.read_csv(fname, sep="|", low_memory=False)
        out.append(claims)
    
    claims = pandas.concat(out)
    claims['to_dt'] = pandas.to_datetime(claims.to_dt)
    claims['from_dt'] = pandas.to_datetime(claims.from_dt)
    
    return(claims)

def filter_2015_cohort(claims):
    st = pandas.to_datetime('2015-01-01')
    en = pandas.to_datetime('2015-06-30')
    # get unique pat_id in date_range (st, en)
    pid_list = claims[(claims.from_dt >= st) & ( claims.from_dt <= en)].pat_id.unique()
    # filter data to track only patients in pid_list
    claims = claims[claims.pat_id.isin(pid_list)].copy()
    
    return(claims)
    
if __name__ == "__main__":

    DATADIR = "../data/"

    # read raw data
    claims = read_raw_data_files(DATADIR)
    
    # filter data
    claims = filter_2015_cohort(claims)
        
    # write data
    claims.to_pickle(f'{DATADIR}/filt_data_v1.pkl')
    
    



