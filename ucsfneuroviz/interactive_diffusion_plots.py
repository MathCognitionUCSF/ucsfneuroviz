import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_average_fa_md(profiles_df, tract):
    """Calculate the average FA and MD for a given tract."""
    tract_data = profiles_df[profiles_df['tractID'] == tract]
    average_fa = tract_data['dti_fa'].mean()
    average_md = tract_data['dti_md'].mean()
    return average_fa, average_md

def calculate_laterality(slcount_df, tract_base):
    """
    Calculate the laterality index for a given tract base name.
    Laterality = (Left - Right) / (Left + Right)
    """
    left_tract = f"{tract_base}_L"
    right_tract = f"{tract_base}_R"
    
    left_count = slcount_df[slcount_df['Unnamed: 0'] == left_tract]['n_streamlines_clean'].values[0]
    right_count = slcount_df[slcount_df['Unnamed: 0'] == right_tract]['n_streamlines_clean'].values[0]
    
    laterality = (left_count - right_count) / (left_count + right_count)
    return laterality

def create_fa_md_plots(profiles_df, tract):
    """Generate box/strip and kde plots for average FA and MD."""
    average_fa, average_md = calculate_average_fa_md(profiles_df, tract)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 3]})
    
    # Box/Strip plot
    sns.boxplot(y=[average_fa, average_md], ax=ax[0], color='lightgray', showfliers=False)
    sns.stripplot(y=[average_fa, average_md], jitter=0.3, size=3, ax=ax[0], alpha=0.6)
    ax[0].scatter(x=[0, 1], y=[average_fa, average_md], color='red', s=50)
    ax[0].set_title(f'Distribution of FA and MD for {tract}')
    ax[0].set_ylabel('Value')
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['Average FA', 'Average MD'])
    
    # KDE plot
    sns.kdeplot([average_fa, average_md], ax=ax[1], shade=True)
    ax[1].set_title(f'KDE of FA and MD for {tract}')
    ax[1].set_xlabel('Value')
    
    plt.tight_layout()
    return fig

def on_submit_button_clicked(button):
    """Action when the 'Submit' button is clicked."""
    metric_type = metric_selector.value
    
    if metric_type == 'Laterality':
        # Display Laterality Plot
        fig = plot_laterality(laterality, tract_base)
        
    elif metric_type == 'FA':
        # Display FA Plot
        fig = create_fa_md_plots(profiles_df, tract, 'FA')
        
    elif metric_type == 'MD':
        # Display MD Plot
        fig = create_fa_md_plots(profiles_df, tract, 'MD')
        
    with out_plot:
        out_plot.clear_output(wait=True)
        display(fig)

# Initialize output widget
out_plot = widgets.Output()

# Radio button for metric type selection
metric_selector = widgets.RadioButtons(
    options=['Laterality', 'FA', 'MD'],
    value='Laterality',
    description='Metric Type:',
    disabled=False
)

# Submit button
submit_button = widgets.Button(
    description='Submit',
    disabled=False,
    button_style='', 
    tooltip='Click to submit',
    icon='check'
)

# Assign the update function to the button
submit_button.on_click(on_submit_button_clicked)

# Interactive table for tract selection
tract_selector = create_interactive_table_with_new_plots(slcount_df, profiles_df, out_plot)

# Display widgets and outputs
display(widgets.VBox([metric_selector, submit_button]))
display(widgets.HBox([tract_selector, out_plot]))
