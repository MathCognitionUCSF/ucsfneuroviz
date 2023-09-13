# write function to import dataframe from csv file and return a dataframe
import pandas as pd
import csv

#%%
def import_dataframe(filename):
    """
    Import a csv file and return a dataframe
    """
    df = pd.read_csv(filename)
    return df

def read_csv_as_list(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data