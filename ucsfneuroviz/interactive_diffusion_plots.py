import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import widgets, HBox, VBox, Output
from IPython.display import display
import glob
import os
import regex as re

# Simulate data
# ... (Same as the earlier code for simulating slcount_df and profiles_df)
def create_dti_metric_df(slcount_path, metric_keyword):
    """Read DTI metric CSV files from a directory and its subdirectories."""
    # Get all csv files with the streamlines count data inside any subfolder in the specified path
    slcount_files = glob.glob(os.path.join(slcount_path, '**', f'*{metric_keyword}*.csv'), recursive=True)
    dfs = []

    # slcount_df = pd.DataFrame()
    for f in slcount_files:
        # Extract the subject ID from the csv file name using regex
        subject_id_match = re.search(r'(?<=sub-)\d+', f)

        if subject_id_match:
            subject_id = subject_id_match.group(0)
        else:
            # print(f"Warning: Could not extract subject ID from file {f}. Skipping.")
            continue

        # Read the data and add the subject ID as a column
        subject_df = pd.read_csv(f)
        subject_df['subjectID'] = subject_id
        dfs.append(subject_df)

    display(pd.concat(dfs, ignore_index=True)) # delete later

    return pd.concat(dfs, ignore_index=True)

# Helper functions to calculate average FA, MD, and laterality
def calculate_average_fa_md(profiles_df, tract):
    tract_data = profiles_df[profiles_df['tractID'] == tract]
    average_fa = tract_data['dti_fa'].mean()
    average_md = tract_data['dti_md'].mean()
    return average_fa, average_md

def calculate_laterality(slcount_df, tract_base):
    left_tract = f"{tract_base}_L"
    right_tract = f"{tract_base}_R"
    left_count = slcount_df[slcount_df['Unnamed: 0'] == left_tract]['n_streamlines_clean'].values[0]
    right_count = slcount_df[slcount_df['Unnamed: 0'] == right_tract]['n_streamlines_clean'].values[0]
    laterality = (left_count - right_count) / (left_count + right_count)
    return laterality

# Function to generate box plots and KDE plots
# Function to generate box plots and KDE plots
def generate_plots(slcount_df, profiles_df, tract, sub_id, ses_id):
    tract_base = tract[:-2]  # Remove the _L or _R to get the base name
    
    # Calculate average FA, MD, and laterality
    average_fa, average_md = calculate_average_fa_md(profiles_df, tract)
    laterality = calculate_laterality(slcount_df, tract_base)
    
    # Get the values for the current subject
    subject_fa, subject_md = calculate_average_fa_md(profiles_df[profiles_df['subjectID'] == sub_id], tract)
    subject_laterality = calculate_laterality(slcount_df[slcount_df['subjectID'] == sub_id], tract_base)
    
    # Generate plots
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    
    # Box plots
    metrics = [average_fa, average_md, laterality]
    subject_metrics = [subject_fa, subject_md, subject_laterality]
    sns.boxplot(y=metrics, ax=ax[0, 0], color='lightgray')
    sns.stripplot(y=metrics, jitter=0.3, size=3, ax=ax[0, 0], alpha=0.6)
    ax[0, 0].scatter(x=[0, 1, 2], y=subject_metrics, color='red', s=50)
    ax[0, 0].set_title('Box Plots')
    ax[0, 0].set_ylabel('Value')
    ax[0, 0].set_xticks([0, 1, 2])
    ax[0, 0].set_xticklabels(['Average FA', 'Average MD', 'Laterality'])
    
    # KDE plots
    sns.kdeplot([average_fa], ax=ax[1, 0], label='Average FA', shade=True)
    sns.kdeplot([average_md], ax=ax[1, 0], label='Average MD', shade=True)
    sns.kdeplot([laterality], ax=ax[1, 0], label='Laterality', shade=True)
    ax[1, 0].axvline(x=subject_fa, color='r', linestyle='--')
    ax[1, 0].axvline(x=subject_md, color='r', linestyle='--')
    ax[1, 0].axvline(x=subject_laterality, color='r', linestyle='--')
    ax[1, 0].set_title('KDE Plots')
    ax[1, 0].set_xlabel('Value')
    ax[1, 0].legend()
    
    plt.tight_layout()
    return fig


# Interactive function to update plots based on selected tract
def create_interactive_table_with_new_plots(slcount_df, profiles_df, out_plot, sub_id, ses_id):
    tract_list = slcount_df['Unnamed: 0'].unique()
    tract_selector = widgets.Select(options=tract_list, description='Tract:', rows=25)
    tract_selector.layout.width = '400px'
  
    def on_tract_selected(change, sub_id, ses_id):
        tract = change['new']
        fig = generate_plots(slcount_df, profiles_df, tract, sub_id, ses_id)
        with out_plot:
            out_plot.clear_output(wait=True)
            display(fig)

    tract_selector.observe(on_tract_selected, names='value')
    
    # Trigger the initial plot
    on_tract_selected({'new': tract_list[0]})
    
    return tract_selector

def interactive_dti_metrics(proc_output_dir, sub_id, ses_id):
    # Initialize output widget
    out_new_plot = widgets.Output()

    # Get all the data into dataframes
    slcount_df = create_dti_metric_df(proc_output_dir, "slCount")
    profiles_df = create_dti_metric_df(proc_output_dir, "profiles")

    # Generate the interactive table and plots
    tract_selector = create_interactive_table_with_new_plots(slcount_df, profiles_df, out_new_plot, sub_id, ses_id)

    # Display widgets
    display(widgets.HBox([tract_selector, out_new_plot]))

