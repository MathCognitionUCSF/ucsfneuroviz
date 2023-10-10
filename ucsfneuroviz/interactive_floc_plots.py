import os
import ipywidgets as widgets
from IPython.display import display, IFrame, HTML

def get_contrast_name(dropdown_val):
    # Define the contrast name based on the dropdown value
    if dropdown_val == 'Faces - Other':
        contrast_name = 'faces-oth'
    elif dropdown_val == 'Words - Numbers':
        contrast_name = 'words-num'
    elif dropdown_val == 'Faces - Places':
        contrast_name = 'faces-places'
    else:
        contrast_name = 'faces-oth'
    return contrast_name

def plot_floc_html(local_path, ldrive_path_func, subject_id, date, p=0.05, contrast_name="faces-oth"):

    file_name_func = f"sub-{subject_id}_ses-{date}_p{p}_interactive_floc_contrast_{contrast_name}.html"
    full_path_to_file_func = os.path.join(ldrive_path_func, contrast_name, file_name_func)

    # if file is not already in local folder, copy it from ldrive
    if not os.path.isfile(os.path.join(local_path, file_name_func)):
        os.system(f"cp {full_path_to_file_func} {local_path}")

    full_path_local_func = os.path.join(local_path, file_name_func)

    iframe = IFrame(src=full_path_local_func, width="100%", height="600")
    display(iframe)

# create 2 dropdown widgets to selected the first and second elements of the contrast name, and a slider to choose the p-value (0.001, 0.01, 0.05)
def interactive_floc_page(local_path, ldrive_path_func, subject_id, date):
    # Define the dropdown widgets
    contrast_dropdown = widgets.Dropdown(
        options=['Faces - Other', 'Words - Numbers', 'Faces - Places'],
        value='Faces - Other',
        description='Contrast Type:',
        disabled=False,
    )
    # Slider for p-value
    p_slider = widgets.SelectionSlider(
        options=[0.001, 0.01, 0.05],
        value=0.001,
        description='p-value:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True
    )
    
    out_html = widgets.Output()

    def update_plot(change):
        with out_html:
            out_html.clear_output()
            contrast_name = get_contrast_name(contrast_dropdown.value)
            plot_floc_html(local_path, ldrive_path_func, subject_id, date, p=p_slider.value, contrast_name=contrast_name)

    # Attach observer to the slider
    p_slider.observe(update_plot, names='value')

    # Make the plot update when the dropdown value changes
    contrast_dropdown.observe(update_plot, names='value')
    
    # Initialize the plot with default values
    update_plot(None)
    
    display(HTML(f'<h3 style="color: #052049;">Select the contrast and threshold for the Functional Localizer task.<br></h3>'))
    display(widgets.HBox([contrast_dropdown, p_slider]))
    display(out_html)

            
