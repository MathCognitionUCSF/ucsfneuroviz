import pandas as pd

def extract_dc_diagnoses(df):
    diagnoses = []
    for col in df.columns:
        if 'Dyslexia Center Diagnosis: (choice=' in col:
            diagnosis = col.split('=')[-1]
            # replace the very last parenthesis with nothing
            diagnosis = diagnosis[::-1].replace(')', '', 1)[::-1]
            # remove the stuff in curly braces
            # diagnosis = diagnosis.split('{')[0].strip()
            diagnoses.append(diagnosis)
    return diagnoses