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
from ucsfneuroviz.utils import extract_dc_diagnoses

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
def get_subject_dc_diagnosis(id_number, behavior_df):
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
        primary_phenotype = behavior_df[behavior_df['ID Number'] == id_number]['Dyslexia Phenotype'].values[0]
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
    elif int(id_number_str) not in df['ID Number'].values:
        with out_error:
            out_error.clear_output(wait=True)
            display(HTML('<h3 style="color: red;">Error: The ID number you entered does not exist.</h3>'))
        return None
    else:
        with out_error:
            out_error.clear_output(wait=True)  # Clear the error if the new ID is valid
        return np.int64(id_number_str)



def zscore_subject(id_number, brain_df, behavior_df, col, value):
    """
    Returns:
    z-score for each region of the subject with the given ID number compared to the mean and standard deviation of the given diagnosis group
    compare_brain_data: brain data for the given diagnosis group"""
    # Construct the column name based on diagnosis
    if col == 'All Children':
        compare_brain_data = brain_df.copy()
    elif col == 'Dyslexia Center Diagnosis':
        # Get subjects from the behavioral dataframe based on the selected diagnosis
        compare_subjects = behavior_df[behavior_df['Dyslexia Center Diagnosis: (choice=' + value + ')'] == "Checked"]['ID Number'].tolist()
        # Filter the brain data dataframe based on these subjects
        compare_brain_data = brain_df[brain_df['id_number'].isin(compare_subjects)]
        
    else:
        # Get subjects from the behavioral dataframe based on the selected diagnosis
        compare_subjects = behavior_df[behavior_df[col] == value]['ID Number'].tolist()
    
        # Filter the brain data dataframe based on these subjects
        compare_brain_data = brain_df[brain_df['id_number'].isin(compare_subjects)]  
    
    # Ensure we exclude the patient data from compare data
    compare_brain_data = compare_brain_data[compare_brain_data['id_number'] != id_number].drop(columns=['id_number'])
    
    # Compute z-score as before
    patient_df = brain_df[brain_df['id_number'] == id_number].drop(columns=['id_number'])
    compare_means = np.mean(compare_brain_data, axis=0)
    compare_stds = np.std(compare_brain_data, axis=0)
    zscores = pd.DataFrame((patient_df - compare_means) / compare_stds)

    return zscores, compare_brain_data

def plot_hemisphere(hemi, z_data, global_vmin, global_vmax, title, cmap, symmetric_cmap=True):


    """Plots a hemisphere based on the given data."""
    z_data_tmp = z_data.copy()

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
        
    # Fetching Destrieux atlas and fsaverage
    labels, _, names = nib.freesurfer.read_annot(HEMI_CONFIG[hemi]['annot'])
    # Decode the names so we can match them up with the z_data
    names_string = [name.decode('utf-8') for name in names]
    names_dict = dict(zip(names_string, names))
    # Exclude "unknown" and "corpuscallosum" from names_string
    names_string = [name for name in names_string if name not in ["unknown", "corpuscallosum"]] 
    
    # print("names_dict")
    # print(names_dict) #!

    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

    z_data_tmp = z_data_tmp.filter(regex=f'^{HEMI_CONFIG[hemi]["label"]}')
    z_data_tmp.columns = z_data_tmp.columns.str.replace(f'{HEMI_CONFIG[hemi]["label"]}', '')
    # Remove the ending after the _ in the column names, i.e. _thick
    z_data_tmp.columns = z_data_tmp.columns.str.split('_').str[0]
    
    # print("z_data_tmp.columns")
    # print(z_data_tmp.columns) #!

    # Zip the values of z_data whose column name matches the atlas name for each region
    region_values = {name: z_data_tmp[name].values for name in names_string}
    # Now replace the name_string with names original names using the names_dict we created earlier
    region_values = {names_dict[name]: values for name, values in region_values.items()}

    # print("region_values")
    # print(region_values) #!

    # Initialize an array with zeros
    mapped_values = np.zeros_like(labels, dtype=float)

    # Then later in your code, when mapping regions to values
    for name, value in region_values.items():
        region_idx = names.index(name)
        mapped_values[labels == region_idx] = value

    # print("mapped_values")
    # print(mapped_values) #!
    # print(mapped_values.shape)
        
    # Create the surface plot
    if symmetric_cmap==True:
        view = plotting.view_surf(getattr(fsaverage, HEMI_CONFIG[hemi]['fsavg']), mapped_values,
                                cmap=cmap, symmetric_cmap=True,
                                vmax=np.max([np.abs(global_vmin), np.abs(global_vmax)]),
                                title=title)
    else:
        view = plotting.view_surf(getattr(fsaverage, HEMI_CONFIG[hemi]['fsavg']), mapped_values,
                        cmap=cmap, symmetric_cmap=False,
                        vmin = global_vmin, 
                        vmax = global_vmax,
                        title=title)
    
    return widgets.HTML(view.get_iframe())  # Return the widget

def refactored_plot_brain(z_data):
    """Plots the brain using the given z-score data."""
    # Determine the global min and max values across both hemispheres for consistent color scaling
    global_vmin = min(z_data.min())
    global_vmax = max(z_data.max())
    
    left_hemi_widget = plot_hemisphere('left', z_data, global_vmin, global_vmax, title, cmap='coolwarm', symmetric_cmap=True)
    right_hemi_widget = plot_hemisphere('right', z_data, global_vmin, global_vmax, title, cmap='coolwarm', symmetric_cmap=True)

    return widgets.HBox([left_hemi_widget, right_hemi_widget])

def plot_bar_for_thresholded_regions(z_data, dtype, thresh):
    """Plot bar chart for regions with Z-scores above the threshold."""
    prominent_regions = [col for col in z_data.columns if z_data[col].abs().mean() > thresh]
    prominent_regions = sorted(prominent_regions, key=lambda x: z_data[x].mean(), reverse=True)

    fig, ax = plt.subplots(figsize=(15, 5))
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

def interactive_brain_zscore_plot(brain_df, behavior_df, diagnosis_columns, subject_id, date):

    display(HTML(f'<h3 style="color: #052049;">Plot z-scores on a cortical surface comparing the current participant to a selected group of participants.<br></h3>'))
    activate_selected_font('EB Garamond', 'EBGaramond-Regular.ttf')

    # Output widgets
    out_brain = widgets.Output()
    out_table = widgets.Output()
    out_bar = widgets.Output()
    out_bar.add_class('bar-plot-container')
    out_plot = widgets.Output()
    # out_error = widgets.Output()
    
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

    # Radio buttons for data type selection
    data_type_selector = widgets.RadioButtons(
        options=['Gray Matter Volume', 'Surface Area', 'Average Cortical Thickness', 'Local Gyrification Index'],
        description='Structural Metric:',
        disabled=False,
        value='Gray Matter Volume',
        # make it narrower so it doesn't take up the whole screen
        # layout={'width': 'max-content'}
        layout=widgets.Layout(padding='0 0 0 15px', width='max-content')
    )

    # Button to trigger the brain plotting
    plot_brain_button = widgets.Button(
        description='Plot Brain',
        disabled=False,
        button_style='', 
        tooltip='Click to plot z-scores on the brain',
        icon='check'
    )

    thresh_slider = widgets.FloatSlider(
        value=1,
        min=0,
        max=2.0,
        step=0.25,
        description='Threshold:',
        continuous_update=False  # Update only when mouse is released
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

    # Update function
    def update_brain_plot(button):

        out_brain.clear_output(wait=True) # Clear the output widget
        dtype = data_type_selector.value
        brain_df_current = get_dataframe(brain_df, dtype)
        # id_number = validate_id_number(subject_id, brain_df_current, out_error)
        # diagnosis = diagnosis_dropdown.value

        z_data, compare_brain_data = zscore_subject(subject_id, brain_df_current, behavior_df, diagnosis_type_dropdown.value, diagnosis_dropdown.value)
        brain_widget = refactored_plot_brain(z_data)

        DC_diagnoses, primary_phenotype, other_note = get_subject_dc_diagnosis(subject_id, behavior_df)
        print_diagnoses = ', '.join(DC_diagnoses)
        if 'Dyslexia' in print_diagnoses:
            # Insert "(primary phenotype: primary_phenotype)" after "Dyslexia"
            print_diagnoses = print_diagnoses.replace('Dyslexia', f'Dyslexia (primary phenotype: {primary_phenotype})')
        if 'Other' in print_diagnoses:
            # Insert "(diagnosis_other_note)" after "Other"
            print_diagnoses = print_diagnoses.replace('Other', f'Other (note: {other_note})')

        with out_brain:
            # display(HTML(f'<h3 style="color: #878D96;">{subject_id} Dyslexia Center Diagnosis:<br>{print_diagnoses}</h3>')) # UNCOMMENT THIS AND DEBUG DIAGNOSIS PRINTING WHEN READY!!!!
            # display(Markdown(f'    {id_number} Dyslexia Center Diagnosis: {print_diagnoses}'))
            # display(behavior_df[behavior_df['ID Number'] == id_number]['Other:']) ###
            display(brain_widget)

    # Function to update threshold plots
    def update_thresh_plots():
        dtype = data_type_selector.value
        brain_df_current = get_dataframe(brain_df, dtype)
        thresh_value = thresh_slider.value
        z_data, compare_brain_data = zscore_subject(subject_id, brain_df_current, behavior_df, diagnosis_type_dropdown.value, diagnosis_dropdown.value)
        
        # Display the bar plot
        barplot = plot_bar_for_thresholded_regions(z_data, dtype, thresh_value)
        with out_bar:
            out_bar.clear_output(wait=True)
            plt.show(barplot)
            
        # Display the interactive table and the initial box plot and kde plot
        region_selector = create_interactive_table(brain_df_current, compare_brain_data, subject_id, z_data, dtype, thresh_value, out_plot)
        with out_table:                          
            out_table.clear_output(wait=True)
            display(widgets.HBox([region_selector, out_plot]))

    # Function to handle slider change
    def on_slider_value_change(change):
        update_thresh_plots()

    # Assign the update function to the button
    plot_brain_button.on_click(update_brain_plot)
    # Assign the update function to the slider
    thresh_slider.observe(on_slider_value_change, names='value')

    # Step 1: Display the static widgets first
    display(widgets.HBox([diagnosis_type_dropdown, diagnosis_dropdown, data_type_selector, plot_brain_button]))

    # Step 2: Trigger the initial brain plot
    update_brain_plot(None)

    # Step 3: Display the output areas
    display(out_brain)
    display(thresh_slider)
    display(widgets.VBox([out_bar]))
    display(out_table)

    # Step 4: Activate dynamic content
    on_slider_value_change({'new': 1})