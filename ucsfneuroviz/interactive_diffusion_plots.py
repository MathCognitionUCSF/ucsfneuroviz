
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from ipywidgets import widgets, HBox, VBox, Output
from IPython.display import display, IFrame, HTML
import glob
import os
import regex as re

from ucsfneuroviz.utils import extract_dc_diagnoses

def create_dti_metric_df(slcount_path, metric_keyword):
    """Read DTI metric CSV files from a directory and its subdirectories."""
    # Get all csv files with the streamlines count data inside any subfolder in the specified path
    slcount_files = glob.glob(os.path.join(slcount_path, '**', f'*{metric_keyword}*.csv'), recursive=True)

    # print(f"Found {len(slcount_files)} files with {metric_keyword} data.") # delete later

    dfs = []
    for f in slcount_files:
        # Extract the subject ID from the csv file name using regex
        subject_id = re.search(r'(?<=sub-)\d+', f).group(0)
        if metric_keyword == "slCount":

            # Read the data and add the subject ID as a column
            subject_df = pd.read_csv(f)
            # Name the Unnamed: 0 column to tractID
            subject_df = subject_df.rename(columns={'Unnamed: 0': 'tractID'})
            # (Doesn't have first col as numbered index and second col is unnamed)
        
        elif metric_keyword == "profiles":
            # Read the data and add the subject ID as a column
            subject_df = pd.read_csv(f, index_col=0)
            # (Already has first col as numbered index and second col as tractID)

        subject_df['id_number'] = subject_id

        # display(subject_df) # delete later

        dfs.append(subject_df)

    # display(pd.concat(dfs, ignore_index=True)) # delete later

    return pd.concat(dfs, ignore_index=True)

# Helper functions to calculate average FA, MD, and laterality
def calculate_average_fa_md(profiles_df, tract):
    tract_data = profiles_df[profiles_df['tractID'] == tract]
    average_fa = tract_data['dti_fa'].mean()
    average_md = tract_data['dti_md'].mean()
    return average_fa, average_md

def calculate_laterality(slcount_df, tract):

    # If the tract name ends in _L, then the left tract is the tract name and the right tract is the tract name with _L replaced with _R
    if tract.endswith('_L'):
        left_tract = tract
        right_tract = tract.replace('_L', '_R')
    # If the tract name ends in _R, then the right tract is the tract name and the left tract is the tract name with _R replaced with _L
    elif tract.endswith('_R'):
        left_tract = tract.replace('_R', '_L')
        right_tract = tract
    else:
        return None

    left_count = slcount_df[slcount_df['tractID'] == left_tract]['n_streamlines_clean'].values[0]
    right_count = slcount_df[slcount_df['tractID'] == right_tract]['n_streamlines_clean'].values[0]
    laterality = (left_count - right_count) / (left_count + right_count)

    return laterality


def get_metric_per_subject(data_df, metrict_type):
    # Takes in a dataframe of DTI metrics and returns the list of values for each subject in the dataframe
    # Assumes that the dataframe has a column called 'id_number' that contains the subject ID and a column called 'tractID' that contains the tract name
    # Uses the calculate_average_fa_md and calculate_laterality functions to calculate the metric for each subject
    # Returns a list of values for each subject in the dataframe

    # Get the list of tracts in the dataframe
    tract_list = data_df['tractID'].unique()

    # Initialize the list of values for each subject
    metric_dict = {}

    # Iterate through each tract and calculate the metric for each subject
    for tract in tract_list:
        metric_dict[f"{tract} {metrict_type}"] = []
        for sub_id in data_df['id_number'].unique():
            df = data_df[data_df['id_number']==sub_id]
            if metrict_type == 'Average FA':
                metric = calculate_average_fa_md(df, tract)[0]
            elif metrict_type == 'Average MD':
                metric = calculate_average_fa_md(df, tract)[1]
            elif metrict_type == 'Laterality':
                metric = calculate_laterality(df, tract)
            else:
                print('Error: metric type not recognized')
                return None
            metric_dict[f"{tract} {metrict_type}"].append(metric)
              
    return metric_dict

def get_sub_and_comparison_data(sub_id, dti_data, behavior_df, col, value):
    """
    Returns:
    sub_data: data for the given subject
    compare_data: data for the given diagnosis group"""

    # Convert id_number to np.int64() to match the type of sub_id
    dti_data['id_number'] = dti_data['id_number'].astype(np.int64)

    # Construct the column name based on diagnosis
    if col == 'All Children':
        compare_data = dti_data.copy()
    elif col == 'Dyslexia Center Diagnosis':
        # Get subjects from the behavioral dataframe based on the selected diagnosis
        compare_subjects = behavior_df[behavior_df['Dyslexia Center Diagnosis: (choice=' + value + ')'] == "Checked"]['ID Number'].tolist()
        # Filter the brain data dataframe based on these subjects
        compare_data = dti_data[dti_data['id_number'].isin(compare_subjects)]
        
    else:
        # Get subjects from the behavioral dataframe based on the selected diagnosis
        compare_subjects = behavior_df[behavior_df[col] == value]['ID Number'].tolist()
    
        # Filter the brain data dataframe based on these subjects
        compare_data = dti_data[dti_data['id_number'].isin(compare_subjects)]  
    
    # Ensure we exclude the patient data from compare data
    compare_data = compare_data[compare_data['id_number'] != sub_id]
    
    # Compute z-score as before
    sub_data = dti_data[dti_data['id_number'] == sub_id]

    # print(f"Sub data: {sub_data}") # delete later
    # print(f"Compare data: {compare_data}") # delete later

    return sub_data, compare_data

# Function to generate box plots and KDE plots
# Function to generate box plots and KDE plots
def generate_plots(sub_data_profiles, compare_data_profiles, sub_data_slcount, compare_data_slcount, tract):

     # Calculate average FA, MD, and laterality for the subject and the comparison group
    FA_dict_compare = get_metric_per_subject(compare_data_profiles, 'Average FA')
    MD_dict_compare = get_metric_per_subject(compare_data_profiles, 'Average MD')
    laterality_dict_compare = get_metric_per_subject(compare_data_slcount, 'Laterality')

    FA_dict_sub = get_metric_per_subject(sub_data_profiles, 'Average FA')
    MD_dict_sub = get_metric_per_subject(sub_data_profiles, 'Average MD')
    laterality_dict_sub = get_metric_per_subject(sub_data_slcount, 'Laterality')

    # Extract the metrics for the specific tract
    compare_fa = FA_dict_compare.get(f"{tract} Average FA", None)
    compare_md = MD_dict_compare.get(f"{tract} Average MD", None)
    compare_laterality = laterality_dict_compare.get(f"{tract} Laterality", None)
    
    subject_fa = FA_dict_sub.get(f"{tract} Average FA", None)
    # print(f"Subject FA: {subject_fa}") # delete later
    subject_md = MD_dict_sub.get(f"{tract} Average MD", None)
    subject_laterality = laterality_dict_sub.get(f"{tract} Laterality", None)
    
    # Generate plots
    fig, ax = plt.subplots(3, 2, figsize=(7, 7), gridspec_kw={'width_ratios': [2, 3]})
    
    metrics_list = [(compare_fa, subject_fa, 'FA', 'dti_fa'), (compare_md, subject_md, 'MD', 'dti_md'), (compare_laterality, subject_laterality, 'Laterality', None)]
    
    for i, (compare_metric, subject_metric, label, tract_metric_column) in enumerate(metrics_list):
        # Boxplot
        sns.boxplot(y=compare_metric, ax=ax[i, 0], color='lightgray', width=0.5)
        sns.stripplot(y=compare_metric, jitter=0.3, size=3, ax=ax[i, 0], alpha=0.6, s=5)
        if subject_metric is not None:  # Add this check to avoid plotting None
            ax[i, 0].scatter(x=0, y=subject_metric, color='red', s=50)
        # Make this plot narrower
        ax[i, 0].set_title(f'Distribution of Average {label}')
        ax[i, 0].set_xlabel('Subjects')
        ax[i, 0].set_ylabel(label)

        # Line Plot for FA and MDf
        if tract_metric_column:
            subject_tract_data = sub_data_profiles[sub_data_profiles['tractID'] == tract][tract_metric_column].values.tolist()
            
            # Using nodeID and tractID as the index, calculate the mean and standard error of the mean for each node across subjects
            compare_tract_data = compare_data_profiles[compare_data_profiles['tractID'] == tract]
            mean_compare = compare_tract_data.groupby(['nodeID', 'tractID']).mean()[tract_metric_column].values.tolist()
            sem_compare = compare_tract_data.groupby(['nodeID', 'tractID']).sem()[tract_metric_column].values.tolist()

            # plot a line plot with shading for the standard error of the mean
            ax[i, 1].plot(mean_compare, color='lightgray')
            # get the x values for the shading
            lower_shade = np.array(mean_compare) - np.array(sem_compare)
            upper_shade = np.array(mean_compare) + np.array(sem_compare)
            ax[i, 1].fill_between(np.arange(len(mean_compare)), lower_shade, upper_shade, color='lightgray', alpha=0.5)
            # plot the subject's data on top
            ax[i, 1].plot(subject_tract_data, color='red')
            ax[i, 1].set_title(f'{label} along {tract}')
            ax[i, 1].set_ylabel(label)
            ax[i, 1].set_xlabel('Node')
            # Create custom legend handles
            handle_comp_group = mlines.Line2D([], [], color='lightgray', label='Comparison Group')
            handle_subject = mlines.Line2D([], [], color='red', label='Subject')

            # Apply custom legend
            ax[i, 1].legend(handles=[handle_comp_group, handle_subject], loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            # make a bar plot for laterality using the streamlines count data
            # get left and right slcount for the subject using sub_data_slcount
            # get tract_base by removing the _L or _R from the tract name if it exists
            if tract.endswith('_L'):
                tract_base = tract.replace('_L', '')
            elif tract.endswith('_R'):
                tract_base = tract.replace('_R', '')

            left_slcount = sub_data_slcount[sub_data_slcount['tractID'] == tract_base + '_L']['n_streamlines_clean'].values[0]
            right_slcount = sub_data_slcount[sub_data_slcount['tractID'] == tract_base + '_R']['n_streamlines_clean'].values[0]

            ax[i, 1].bar(x=['Left', 'Right'], height=[left_slcount, right_slcount], color=['lightgray', 'lightgray'])
            ax[i, 1].set_title(f'Streamline Counts for {tract_base}')
            ax[i, 1].set_ylabel('Streamlines')
            ax[i, 1].set_xlabel('Hemisphere')

    plt.tight_layout()
    return fig


# Interactive function to update plots based on selected tract
def create_interactive_table_with_new_plots(sub_data_profiles, compare_data_profiles, sub_data_slcount, compare_data_slcount, out_plot, sub_id, ses_id):

    # display(slcount_df) # delete later
    # print(slcount_df['tractID'].unique()) # delete later
    # display(sub_data_slcount) # delete later

    # tract_list = sub_data_slcount['tractID'].unique().tolist()

    # Load in tract_key.csv from data
    tract_key = pd.read_csv('./data/tract_key.csv')
    # Zip labels column and tractID column into a dictionary
    tract_key_dict = dict(zip(tract_key['label'], tract_key['tractID']))

    # print(f"Tract list: {tract_list}")  # Debug print
    tract_selector = widgets.Select(options=list(tract_key_dict.keys()), description='Tract:', rows=25)
    tract_selector.layout.width = '400px'
  
    def on_tract_selected(change):
        tract_label = change['new']
        tract = tract_key_dict[tract_label]
        # Adding debug prints to understand the flow
        # print(f"Selected Tract: {tract}")
        
        fig = generate_plots(sub_data_profiles, compare_data_profiles, sub_data_slcount, compare_data_slcount, tract)
        with out_plot:
            out_plot.clear_output(wait=True)
            # print("Displaying the plot...")  # Debug print
            display(fig)

    # Using a lambda to pass extra arguments
    tract_selector.observe(lambda change: on_tract_selected(change), names='value')
    
    # Trigger the initial plot and add debug prints
    # print("Triggering the initial plot...")  # Debug print
    first_value = list(tract_key_dict.keys())[0]
    on_tract_selected({'new': first_value})
    
    return tract_selector

def interactive_dti_metrics(slcount_df, profiles_df, behavior_df, diagnosis_columns, sub_id, ses_id):
    # Dropdown for diagnosis type selection
    diagnosis_type_dropdown = widgets.Dropdown(
        options=['All Children', 'Dyslexia Center Diagnosis'] + diagnosis_columns,  # Add other diagnosis types here
        description='Diagnosis Type:',
        value='All Children',  # Set default value
        disabled=False,
    )

    # Dropdown for diagnosis selection
    diagnosis_dropdown = widgets.Dropdown(
        options=['All Children'],  # Initially, only "All Children" is available
        description='Compare to:',
        value='All Children',  # Set default value
        disabled=False
    )

    # Submit button
    submit_button = widgets.Button(
        description='Submit',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Submit',
        icon='check'
    )

    def update_diagnosis_options(change):
        new_type = change['new']
        if new_type == 'All Children':
            diagnosis_dropdown.options = ['All Children']
            diagnosis_dropdown.value = 'All Children'
        elif new_type == 'Dyslexia Center Diagnosis':
            unique_vals = extract_dc_diagnoses(behavior_df)
            diagnosis_dropdown.options = unique_vals
            diagnosis_dropdown.value = unique_vals[0] if unique_vals else None
        else:
            unique_vals = list(behavior_df[new_type].dropna().unique())
            diagnosis_dropdown.options = unique_vals
            diagnosis_dropdown.value = unique_vals[0] if unique_vals else None

    # Observe changes in diagnosis type dropdown and update diagnosis options accordingly
    diagnosis_type_dropdown.observe(update_diagnosis_options, names='value')

    # Initialize output widget
    out_plot = widgets.Output()

    def update_sub_and_comparison_data(button):  # button parameter to catch the event trigger
        # Clear previous output
        out_plot.clear_output(wait=True)
        
        # Fetch new data
        sub_data_profiles, compare_data_profiles = get_sub_and_comparison_data(
            sub_id, profiles_df, behavior_df, 
            col=diagnosis_type_dropdown.value, 
            value=diagnosis_dropdown.value
        )
        sub_data_slcount, compare_data_slcount = get_sub_and_comparison_data(
            sub_id, slcount_df, behavior_df, 
            col=diagnosis_type_dropdown.value, 
            value=diagnosis_dropdown.value
        )
        
        # Update plots and widgets
        tract_selector = create_interactive_table_with_new_plots(
            sub_data_profiles, compare_data_profiles, 
            sub_data_slcount, compare_data_slcount, 
            out_plot, sub_id, ses_id
        )
        
        # Redisplay the widgets
        display(widgets.HTML("<h3>View metrics for individual tracts compared to selected group.</h3>"))
        display(widgets.HBox([tract_selector, out_plot]))
    
    # Display widgets
    # display markdown "Explore the white matter tracts interactively."
    display(widgets.HBox([diagnosis_type_dropdown, diagnosis_dropdown, submit_button]))

    # link the submit button to the update_sub_and_comparison_data function
    submit_button.on_click(update_sub_and_comparison_data)
    # Trigger intial plot for 'All Children' default dropdown selection
    update_sub_and_comparison_data(None)

def plot_diffusion_html(local_path, ldrive_path_diffusion, subject_id, date):

    file_name_diffusion = f"sub-{subject_id}_ses-{date}_space-T1w_desc-preproc_dwi_space-RASMM_model-probCSD_algo-AFQ_desc-viz_dwi.html"
    full_path_to_file_diffusion = os.path.join(ldrive_path_diffusion, file_name_diffusion)

    # if file is not already in local folder, copy it from ldrive
    if not os.path.isfile(os.path.join(local_path, file_name_diffusion)):
        os.system(f"cp {full_path_to_file_diffusion} {local_path}")

    full_path_local_func = os.path.join(local_path, file_name_diffusion)

    iframe = IFrame(src=full_path_local_func, width="90%", height="600")
    
    display(HTML(f'<h3 style="color: #052049;">Explore the white matter tracts interactively.<br></h3>'))
    display(iframe)