import pandas as pd
import numpy as np
import os
from nilearn import datasets, plotting 
import ipywidgets as widgets
from IPython.display import display, Markdown
from IPython.core.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib

from matplotlib import font_manager

def activate_selected_font(font_name, font_file_name):
    # Get the path of the script
    script_dir = os.path.dirname(os.path.realpath('__file__'))

    # Use the relative path to locate the font in the repo's directory
    font_path = os.path.join(script_dir, f'fonts/{font_file_name}')

    # Add the font
    font_manager.fontManager.addfont(font_path)

    # Use the font
    plt.rcParams['font.family'] = font_name

# Helper functions
def get_subject_DC_diagnosis(id_number, behavior_df):
    """Returns the Dyslexia Center diagnosis, primary phenotype, and other diagnosis notes of the subject with the given ID number."""
    # Get all vars that start with 'Dyslexia Center Diagnosis'
    DC_diagnosis_vars = [col for col in behavior_df.columns if col.startswith('Dyslexia Center Diagnosis')]
    # Get all columns that have a value of 1 for the given ID number
    DC_diagnosis_cols = [col for col in DC_diagnosis_vars if behavior_df[behavior_df['ID Number'] == id_number][col].values[0] == 1]
    # Get all diagnoses from the column names
    DC_diagnoses = [col.split('=')[-1].replace(')', '') for col in DC_diagnosis_cols]
    # Remove  (e.g. LEE-GT) \{diagnosis_other\}
    DC_diagnoses = [diagnosis.replace('(e.g. LEE-GT {diagnosis_other}', '') for diagnosis in DC_diagnoses]
    # If they have dyslexia, get the primary phenotype
    if 'Dyslexia' in DC_diagnoses:
        primary_phenotype = behavior_df[behavior_df['ID Number'] == id_number]['diagnosis_dyslexia_phenotype'].values[0]
    else:
        primary_phenotype = ''
    if 'Other' in DC_diagnoses:
        other_note = behavior_df[behavior_df['ID Number'] == id_number]['Other:'].values[0]
    else:
        other_note = ''
    return DC_diagnoses, primary_phenotype, other_note

def get_dataframe(struct_df, dtype):
    # Create individual dataframes for each measure
    thick_df = struct_df.filter(regex=('_thick$'))
    thick_df.insert(0, 'id_number', struct_df['id_number'])
    vol_df = struct_df.filter(regex='_vol$')
    vol_df.insert(0, 'id_number', struct_df['id_number'])
    area_df = struct_df.filter(regex='_area$')
    area_df.insert(0, 'id_number', struct_df['id_number'])
    lgi_df = struct_df.filter(regex='_lgi$')
    lgi_df.insert(0, 'id_number', struct_df['id_number'])

    """Returns the corresponding dataframe based on the data type."""
    dataframes = {
        'Gray Matter Volume': vol_df,
        'Surface Area': area_df,
        'Average Cortical Thickness': thick_df,
        'Local Gyrification Index': lgi_df
    }
    return dataframes.get(dtype)

def validate_id_number(id_number_str, df, out_error):
    """Validates the ID number input by the user."""
    if not id_number_str.isdigit() or len(id_number_str) != 5:
        with out_error:
            out_error.clear_output(wait=True)
            display(HTML('<h3 style="color: red;">Error: Please enter a 5-digit number for the ID number.</h3>'))
        return None
    elif int(id_number_str) not in df['id_number'].values:
        with out_error:
            out_error.clear_output(wait=True)
            display(HTML('<h3 style="color: red;">Error: The ID number you entered does not exist.</h3>'))
        return None
    else:
        with out_error:
            out_error.clear_output(wait=True)  # Clear the error if the new ID is valid
        return np.int64(id_number_str)

def extract_diagnoses(df):
    diagnoses = []
    for col in df.columns:
        if 'Dyslexia Center Diagnosis: (choice=' in col:
            # diagnosis = col.split('=')[-1].replace(')', '')
            diagnoses.append(diagnosis)
    # Append to the front, so that 'All Children' is the first option
    diagnoses.insert(0, 'All Children')
    return diagnoses

def zscore_subject(id_number, brain_df, behavior_df, col, value):
    """
    Returns:
    z-score for each region of the subject with the given ID number compared to the mean and standard deviation of the given diagnosis group
    compare_brain_data: brain data for the given diagnosis group"""
    # Construct the column name based on diagnosis
    if col != 'All Children':
    
        # Get subjects from the behavioral dataframe based on the selected diagnosis
        compare_subjects = behavior_df[behavior_df[col] == value]['ID Number'].tolist()
    
        # Filter the brain data dataframe based on these subjects
        compare_brain_data = brain_df[brain_df['id_number'].isin(compare_subjects)]

    else:
        compare_brain_data = brain_df.copy()
    
    # Ensure we exclude the patient data from compare data
    compare_brain_data = compare_brain_data[compare_brain_data['id_number'] != id_number].drop(columns=['id_number'])
    
    # Compute z-score as before
    patient_df = brain_df[brain_df['id_number'] == id_number].drop(columns=['id_number'])
    compare_means = np.mean(compare_brain_data, axis=0)
    compare_stds = np.std(compare_brain_data, axis=0)
    zscores = pd.DataFrame((patient_df - compare_means) / compare_stds)
    # display(zscores)
    # display(compare_brain_data)

    return zscores, compare_brain_data

# Configuration for each hemisphere
HEMI_CONFIG = {
    'left': {
        'label': 'lh_',
        'annot': 'data/lh.aparc.annot',
        'fsavg': 'infl_left'
    },
    'right': {
        'label': 'rh_',
        'annot': 'data/rh.aparc.annot',
        'fsavg': 'infl_right'
    }
}

def plot_hemisphere(hemi, z_data, global_vmin, global_vmax):
    """Plots a hemisphere based on the given data."""
    z_data_tmp = z_data.copy()
    
    # Fetching Destrieux atlas and fsaverage
    labels, _, names = nib.freesurfer.read_annot(HEMI_CONFIG[hemi]['annot'])
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

    z_data_tmp = z_data_tmp.filter(regex=f'^{HEMI_CONFIG[hemi]["label"]}')
    z_data_tmp.columns = z_data_tmp.columns.str.replace(f'{HEMI_CONFIG[hemi]["label"]}', '')

    # Zip the values of z_data to the labels of the Destrieux atlas
    region_values = dict(zip(names, z_data_tmp.values[0]))

    # Initialize an array with zeros
    mapped_values = np.zeros_like(labels, dtype=float)

    # Populate the mapped_values array using region_values dictionary
    for label, value in region_values.items():
        region_idx = names.index(label)
        mapped_values[labels == region_idx] = value

    view = plotting.view_surf(getattr(fsaverage, HEMI_CONFIG[hemi]['fsavg']), mapped_values,
                              cmap='coolwarm', symmetric_cmap=True,
                              vmax=np.max([np.abs(global_vmin), np.abs(global_vmax)]))
    return widgets.HTML(view.get_iframe())  # Return the widget

def refactored_plot_brain(z_data):
    """Plots the brain using the given z-score data."""
    # Determine the global min and max values across both hemispheres for consistent color scaling
    global_vmin = min(z_data.min())
    global_vmax = max(z_data.max())
    
    left_hemi_widget = plot_hemisphere('left', z_data, global_vmin, global_vmax)
    right_hemi_widget = plot_hemisphere('right', z_data, global_vmin, global_vmax)

    return widgets.HBox([left_hemi_widget, right_hemi_widget])

def plot_bar_for_thresholded_regions(z_data, dtype, thresh):
    """Plot bar chart for regions with Z-scores above the threshold."""
    prominent_regions = [col for col in z_data.columns if z_data[col].abs().mean() > thresh]
    prominent_regions = sorted(prominent_regions, key=lambda x: z_data[x].mean(), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(prominent_regions, z_data[prominent_regions].values[0], color=(0.2, 0.4, 0.6, 0.6))
    ax.set_ylabel('Z-Score')
    ax.set_title(f'Regions with Abs(Z-Score) > {thresh}')
    ax.set_xticks(prominent_regions)
    ax.set_xticklabels(prominent_regions, rotation=90)
    ax.axhline(y=0, color='black', linestyle='--')
    plt.tight_layout()
    return fig

def create_plot(df, compare_brain_data, id_number, dtype, region):
    """Generate scatter and distribution plots for a specific region."""
    # df = get_dataframe(dtype)
    region_data = compare_brain_data[region]
    subject_data = df[df['id_number'] == id_number][region].values[0]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 3]})
    sns.boxplot(y=region_data, ax=ax[0], color='lightgray', showfliers=False)
    sns.stripplot(y=region_data, jitter=0.3, size=3, ax=ax[0], alpha=0.6)
    ax[0].scatter(x=0, y=subject_data, color='red', s=50, label=f'Subject {id_number}: Val={subject_data:.2f}')
    ax[0].set_title(f'Distribution of {region}')
    ax[0].set_ylabel(dtype)
    ax[0].set_xticks([])
    ax[0].set_xlabel('Subjects')
    ax[0].legend()
    sns.kdeplot(region_data, ax=ax[1], shade=True)
    z_val = (subject_data - region_data.mean()) / region_data.std()
    ax[1].axvline(x=subject_data, color='r', linestyle='--', label=f'Subject {id_number}: Z={z_val:.2f}')
    ax[1].set_title(f'KDE of {region}')
    ax[1].set_xlabel(dtype)
    ax[1].legend()
    plt.tight_layout()
    return fig

def create_interactive_table(df, compare_brain_data, id_number, z_data, dtype, thresh, out_plot):
    """Generate an interactive table for regions with Z-scores above the threshold."""
    prominent_regions = [col for col in z_data.columns if z_data[col].abs().mean() > thresh]
    prominent_regions = sorted(prominent_regions, key=lambda x: z_data[x].mean(), reverse=True)

    region_selector = widgets.Select(options=prominent_regions, description='Region:', rows=25)
    region_selector.layout.width = '400px'
  
    def on_region_selected(change):
        region = change['new']
        fig = create_plot(df, compare_brain_data, id_number, dtype, region)
        with out_plot:
            out_plot.clear_output(wait=True)
            display(fig)

    region_selector.observe(on_region_selected, names='value')

    # Trigger the initial plot
    on_region_selected({'new': prominent_regions[0]})

    return region_selector

def interactive_brain_zscore_plot(brain_df, behavior_df):

    activate_selected_font('EB Garamond', 'EBGaramond-Regular.ttf')

    # Output widgets
    out_brain = widgets.Output()
    out_table = widgets.Output()
    out_bar = widgets.Output()
    out_bar.add_class('bar-plot-container')
    out_plot = widgets.Output()
    out_error = widgets.Output()
    
    # Dropdown for diagnosis selection
    diagnosis_dropdown = widgets.Dropdown(
        options=extract_diagnoses(behavior_df),
        description='Compare to:',
        disabled=False
    )
    
    # Textbox for user to enter ID
    id_input = widgets.Text(
        value='',
        placeholder='Enter ID number',
        description='ID Number:',
        disabled=False,  
        layout={'width': 'max-content'}
    )
    
    # Radio buttons for data type selection
    data_type_selector = widgets.RadioButtons(
        options=['Gray Matter Volume', 'Surface Area', 'Average Cortical Thickness', 'Local Gyrification Index'],
        description='Structural Metric:',
        disabled=False,
        value='Gray Matter Volume',
        # make it narrower so it doesn't take up the whole screen
        layout={'width': 'max-content'}
    )

    # Button to trigger the brain plotting
    plot_brain_button = widgets.Button(
        description='Plot Brain',
        disabled=False,
        button_style='', 
        tooltip='Click to plot z-scores on the brain'
    )

    # Widgets for user inputs
    thresh_input = widgets.FloatText(
        value=1,
        description='Threshold:',
        disabled=False,
        layout={'width': 'max-content'}
    )

    plot_thresh_button = widgets.Button(
        description='Plot Regions',
        disabled=False,
        button_style='', 
        tooltip='Click to plot regions with z-scores above the threshold'
    )

    # Update function
    def update_brain_plot(button):
        out_brain.clear_output(wait=True) # Clear the output widget
        dtype = data_type_selector.value
        brain_df_current = get_dataframe(brain_df, dtype)
        id_number = validate_id_number(id_input.value.strip(), brain_df_current, out_error)
        diagnosis = diagnosis_dropdown.value

        z_data, compare_brain_data = zscore_subject(id_number, brain_df_current, behavior_df, diagnosis)
        brain_widget = refactored_plot_brain(z_data)

        DC_diagnoses, primary_phenotype, other_note = get_subject_DC_diagnosis(id_number, behavior_df)
        print_diagnoses = ', '.join(DC_diagnoses)
        if 'Dyslexia' in print_diagnoses:
            # Insert "(primary phenotype: primary_phenotype)" after "Dyslexia"
            print_diagnoses = print_diagnoses.replace('Dyslexia', f'Dyslexia (primary phenotype: {primary_phenotype})')
        if 'Other' in print_diagnoses:
            # Insert "(diagnosis_other_note)" after "Other"
            print_diagnoses = print_diagnoses.replace('Other', f'Other (note: {other_note})')

        with out_brain:
            display(HTML(f'<h3 style="color: #878D96;">{id_number} Dyslexia Center Diagnosis:<br>{print_diagnoses}</h3>'))
            # display(Markdown(f'    {id_number} Dyslexia Center Diagnosis: {print_diagnoses}'))
            # display(behavior_df[behavior_df['ID Number'] == id_number]['Other:']) ###
            display(brain_widget)

    def on_plot_thresh_button_clicked(button):
        """Action when the 'Plot Threshold' button is clicked."""
        dtype = data_type_selector.value
        brain_df_current = get_dataframe(brain_df, dtype)
        id_number = validate_id_number(id_input.value.strip(), brain_df_current, out_error)
        thresh_value = thresh_input.value
        z_data, compare_brain_data = zscore_subject(id_number, brain_df_current, behavior_df, diagnosis_dropdown.value)
        
        # Display the bar plot
        barplot = plot_bar_for_thresholded_regions(z_data, dtype, thresh_value)
        with out_bar:
            out_bar.clear_output(wait=True)
            plt.show(barplot)
            
        # Display the interactive table and the initial box plot and kde plot
        region_selector = create_interactive_table(brain_df_current, compare_brain_data, id_number, z_data, dtype, thresh_value, out_plot)
        with out_table:                          
            out_table.clear_output(wait=True)
            display(widgets.HBox([region_selector, out_plot]))

    # Assign the update function to the button
    plot_brain_button.on_click(update_brain_plot)

    # Assign the update function to the button
    plot_thresh_button.on_click(on_plot_thresh_button_clicked)

    # Display the widgets
    # display(Markdown('## Enter an ID number, group of comparison subjects, and metric type.'))
    display(widgets.HBox([id_input, diagnosis_dropdown, data_type_selector, plot_brain_button]))
    display(out_error)
    display(out_brain)

    # Display widgets
    # display(Markdown('## Enter a threshold to plot the regions where |z-score| > threshold.'))
    display(widgets.HBox([thresh_input, plot_thresh_button]))
    # display(out_bar)
    display(widgets.VBox([out_bar]))
    display(out_table)  
