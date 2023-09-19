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

def get_redcap_data():
    enter_token = widgets.Text(
        value='',
        placeholder='Enter REDCap API token',
        description='Token:',
        disabled=False
    )
    
    button = widgets.Button(description="Submit")
    out = widgets.Output()

    redcap = None
    redcap_metadata = None
    
    def on_button_clicked(b):
        nonlocal redcap, redcap_metadata
        with out:
            clear_output()
            token = enter_token.value
            # Assume import_data_redcap is a function that returns redcap data
            redcap, redcap_metadata = import_data_redcap(token)
            del token
            print("Data fetched")
    
    button.on_click(on_button_clicked)
    
    display(HBox([enter_token, button]), out)
    
    return lambda: (redcap, redcap_metadata)
