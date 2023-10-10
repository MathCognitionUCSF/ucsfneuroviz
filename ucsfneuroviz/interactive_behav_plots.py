import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown, HTML
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from ucsfneuroviz.importer import import_dataframe, read_csv_as_list
from ucsfneuroviz.interactive_struct_plots import activate_selected_font, validate_id_number, extract_dc_diagnoses
from ucsfneuroviz.fc_vars import FC_vars, FC_vars_select

# Global variable to store the color mapping for each group.
group_color_mapping = {}

def get_distinct_colors(n):
    """
    Returns n equally spaced colors from the HSV colormap in RGBA format.
    
    Parameters:
    - n: Number of colors
    
    Returns:
    - List of RGBA tuples
    """
    cmap = plt.cm.hsv
    hues = np.linspace(0, 1, n + 1, endpoint=False)[:-1]  # Avoid endpoint and take n hues
    return  [cmap(h) for h in hues]

def color_generator(n):
    """Yield colors one by one from a list of n distinct colors."""
    colors = get_distinct_colors(n)
    for color in colors:
        yield color

def plot_avg_scores_by_group_with_variation(df, task_order, groupby_col,
                                            show_sem, show_std, show_all,
                                            title, xlabel, ylabel, legend_title, 
                                            y_range=range(0, 101, 10),
                                            figsize=(27, 15), fontsize=38, save_path=''):
    
    global group_color_mapping
    
    # Create or retrieve the group colors
    unique_groups = sorted(df[groupby_col].unique())
    total_unique_groups = len(unique_groups)
    
    # If group_color_mapping is empty, initialize the color generator
    color_gen = color_generator(total_unique_groups)
    
    for group in unique_groups:
        if group not in group_color_mapping:
            # If the group doesn't have a color assigned yet, give it one
            group_color_mapping[group] = next(color_gen)


    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background color to 50% grey
    ax.set_facecolor("#808080")  # This is the code for 50% grey
    fig.patch.set_facecolor("#808080")  # Setting the figure background as well
    
    # Set grid color to very light grey
    ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.5)
    
    # Set font color to very light grey for axes, title, and tick labels
    ax.tick_params(colors='#D3D3D3')
    ax.xaxis.label.set_color('#D3D3D3')
    ax.yaxis.label.set_color('#D3D3D3')
    ax.title.set_color('#D3D3D3')
    
    means = df[list(task_order.values())+[groupby_col]].groupby(groupby_col).mean().T
    if show_sem:
        variation = df[list(task_order.values())+[groupby_col]].groupby(groupby_col).sem().T
    elif show_std:
        variation = df[list(task_order.values())+[groupby_col]].groupby(groupby_col).std().T
    else:
        variation = None
    
    # If "Show All" is toggled, display individual observations
    if show_all:
        for group in means.columns:
            group_data = df[df[groupby_col] == group][list(task_order.values())].values
            for data in group_data:
                ax.plot(list(task_order.values()), data, alpha=0.3, color=group_color_mapping[group], lw=1)
    
    # Plotting means with markers
    for group in means.columns:
        ax.plot(list(task_order.values()), means[group], marker='o', lw=3, color=group_color_mapping[group], label=f"{group} - n={len(df[df[groupby_col] == group])}")
    
    # Adding shaded region for SEM or STD
    if variation is not None:
        for group in means.columns:
            ax.fill_between(list(task_order.values()), 
                            means[group] - variation[group], 
                            means[group] + variation[group], 
                            color=group_color_mapping[group],
                            alpha=0.2)
    
    ax.legend(title=legend_title, title_fontsize=fontsize-20, 
              fontsize=fontsize-20, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize-6)
    ax.set_ylabel(ylabel, fontsize=fontsize-6)
    plt.xticks(ticks=range(len(task_order)), labels=list(task_order.keys()), rotation=90, fontsize=fontsize-10)
    plt.yticks(ticks=y_range, fontsize=fontsize-10) 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def interactive_line_plot(df, groupby_col, FC_vars):
    unique_groups = sorted(df[groupby_col].unique())
    
    group_selector = widgets.SelectMultiple(
        options=unique_groups,
        value=list(unique_groups),
        description='',
        disabled=False
    )
    
    show_all_toggle = widgets.ToggleButton(
        value=False,
        description='Show All',
        disabled=False,
        button_style='', 
        tooltip='Display All Observations',
        layout=widgets.Layout(width='120px')
    )
    
    sem_toggle = widgets.ToggleButton(
        value=False,
        description='Show SEM',
        disabled=False,
        button_style='', 
        tooltip='Display Standard Error of Mean',
        layout=widgets.Layout(width='120px')
    )
    
    std_toggle = widgets.ToggleButton(
        value=False,
        description='Show STD',
        disabled=False,
        button_style='', 
        tooltip='Display Standard Deviation',
        layout=widgets.Layout(width='120px')
    )
    
    def update_plot(selected_groups, show_sem, show_std, show_all):
        filtered_df = df[df[groupby_col].isin(selected_groups)]
        plot_avg_scores_by_group_with_variation(
            df=filtered_df,
            task_order=FC_vars,
            groupby_col=groupby_col,
            show_sem=show_sem,
            show_std=show_std,
            show_all=show_all,
            title="Mean values for selected groups",
            xlabel="Task",
            ylabel="Mean Value",
            legend_title="Diagnosis"
        )

    group_label = widgets.Label(value='Select Groups:')
    toggles = widgets.VBox([show_all_toggle, sem_toggle, std_toggle], layout=widgets.Layout(align_items='center'))
    left_box = widgets.VBox([group_label, group_selector])
    right_box = widgets.VBox([widgets.Label(value='Display Options:'), toggles])
    ui = widgets.HBox([left_box, right_box])
    
    out = widgets.interactive_output(update_plot, {
        'selected_groups': group_selector,
        'show_sem': sem_toggle,
        'show_std': std_toggle,
        'show_all': show_all_toggle
    })
    
    display(ui, out)


# Function to plot heatmap
def plot_heatmap(df, task_order, groupby_col, selected_group, figsize=(27, 15), fontsize=38):
    # Create a figure with custom size
    plt.figure(figsize=figsize)
    
    # Generate heatmap using seaborn without annotations in cells
    ax = sns.heatmap(df[task_order], cmap="viridis", linewidths=.5, cbar_kws={"shrink": 1, "label": "Score"})
    
    # Title with group name and number of subjects
    n_subjects = len(df)
    plt.title(f"{selected_group} (n={n_subjects})", fontsize=fontsize)
    
    # Adjust y-axis
    plt.ylabel("Individuals", fontsize=fontsize-6)
    ax.set_yticks([])  # Remove y-axis tickmarks
    plt.xlabel("Tasks", fontsize=fontsize-6)
    plt.yticks(fontsize=fontsize-10)
    plt.xticks(fontsize=fontsize-10, rotation=90)
    
    # Adjust colorbar font size and label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize-10)
    cbar.set_label("Score", size=fontsize-6)
    
    plt.tight_layout()

    # Display the plot
    plt.show()

# Interactive function
def interactive_heatmap(df, groupby_col, FC_vars):
    # Get unique group names
    group_names = sorted(df[groupby_col].unique())
    
    # Dropdown for group selection with custom font size
    group_selector = widgets.Dropdown(
        options=group_names,
        value=group_names[0],
        description='Select Group:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    
    # Interactive widget
    @widgets.interact(group=group_selector)
    def update_plot(group):
        filtered_df = df[df[groupby_col] == group]
        plot_heatmap(filtered_df, FC_vars, groupby_col, group)

# Function to plot radar/spider plot
def plot_radar(df, task_order, groupby_col, selected_group, figsize=(8, 8), fontsize=16):

    global group_color_mapping
    
    # Create or retrieve the group colors
    unique_groups = sorted(df[groupby_col].unique())
    total_unique_groups = len(unique_groups)
    
    # If group_color_mapping is empty, initialize the color generator
    color_gen = color_generator(total_unique_groups)
    
    for group in unique_groups:
        if group not in group_color_mapping:
            # If the group doesn't have a color assigned yet, give it one
            group_color_mapping[group] = next(color_gen)

    # Number of variables we're plotting
    num_vars = len(task_order)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Set figure and subplot size
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Set background color for the radar only
    ax.set_facecolor("#808080")
    
    # Helper function to plot data on radar chart
    def add_to_radar(data, color, label):
        values = data[task_order].tolist()
        values += values[:1]  # Complete the loop
        ax.plot(angles + angles[:1], values, color=color, linewidth=2, label=label)

    # Add each feature to the radar chart
    for idx, group in enumerate(df[groupby_col].unique()):
        # Select valid columns and calculate the mean to handle the FutureWarning
        valid_cols = df[df[groupby_col] == group][task_order]
        add_to_radar(valid_cols.mean(), group_color_mapping[group], f"{group} (n={len(valid_cols)})")
    
    # Set the angle, labels and location for each label
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    # Calculate label rotations: start from 90 and rotate continuously back to 90
    label_rotations = np.linspace(90, -270, num_vars).tolist()

    for angle, label, rotation in zip(angles, task_order, label_rotations):
        ha = 'center'
        ax.text(angle, ax.get_rmax() + 30, label, rotation=rotation, ha=ha, va='center', fontsize=fontsize-2, color='black')

    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.grid(color='lightgrey')
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.7, 1.2), fontsize=fontsize-4)

    # Show plot
    plt.show()
    
def interactive_radar(df, groupby_col, FC_vars):
    # Get unique group names
    group_names = sorted(df[groupby_col].unique())
    
    # Dropdown for group selection
    group_selector = widgets.SelectMultiple(
        options=group_names,
        value=[group_names[0]],  # Default to the first group
        description='Select Group(s):',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    
    # Interactive widget
    @widgets.interact(group=group_selector)
    def update_plot(group):
        filtered_df = df[df[groupby_col].isin(group)]
        plot_radar(filtered_df, FC_vars, groupby_col, group)

def get_key(my_dict, value):
    for key, val in my_dict.items():
        if val == value:
            return key
    return None

def create_plot(df, compare_behav_data, id_number, FC_vars, task):
    """Generate scatter and distribution plots for a specific region."""
    # df = get_dataframe(dtype)
    task_data = compare_behav_data[task]
    subject_data = df[df['ID Number'] == id_number][task].values[0]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 3]})
    sns.boxplot(y=task_data, ax=ax[0], color='lightgray', showfliers=False)
    sns.stripplot(y=task_data, jitter=0.3, size=3, ax=ax[0], alpha=0.6)
    ax[0].scatter(x=0, y=subject_data, color='red', s=50, label=f'Subject {id_number}: Val={subject_data:.2f}')
    ax[0].set_title(f'Distribution of {get_key(FC_vars, task)}')
    ax[0].set_ylabel('Percentile')
    ax[0].set_xticks([])
    ax[0].set_xlabel('Subjects')
    ax[0].legend()
    sns.kdeplot(task_data, ax=ax[1], shade=True)
    z_val = (subject_data - task_data.mean()) / task_data.std()
    ax[1].axvline(x=subject_data, color='r', linestyle='--', label=f'Subject {id_number}: Z={z_val:.2f}')
    ax[1].set_title(f'KDE of {get_key(FC_vars, task)}')
    ax[1].set_xlabel('Percentile')
    ax[1].legend()
    plt.tight_layout()
    return fig

def create_interactive_table(df, compare_behav_data, id_number, FC_vars,  out_plot):
    """Generate an interactive table for regions with Z-scores above the threshold."""
    
    # Create a list of tuples with keys as labels and values as the actual values
    options_list = [(key, value) for key, value in FC_vars.items()]
    
    task_selector = widgets.Select(options=options_list, description='Region:', rows=25)
    task_selector.layout.width = '400px'
  
    def on_task_selected(change):
        task = change['new']  # This will now be the value, not the key
        fig = create_plot(df, compare_behav_data, id_number, FC_vars, task)
        with out_plot:
            out_plot.clear_output(wait=True)
            display(fig)

    task_selector.observe(on_task_selected, names='value')

    # Trigger the initial plot using the first value from the dictionary
    on_task_selected({'new': list(FC_vars.values())[0]})

    return task_selector

def interactive_individual_line_plot(df, id_col, diagnosis_columns, FC_vars, subject_id, date):
    
    display(HTML(f'<h3 style="color: #052049;">Plot percentiles for behavioral tasks comparing the current participant to a selected group of participants.<br></h3>'))
    activate_selected_font('EB Garamond', 'EBGaramond-Regular.ttf')

    # Create an Output widget
    out_line = widgets.Output()
    out_table = widgets.Output()
    out_plot = widgets.Output()

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
    
    # Toggle buttons for SEM and STD
    sem_toggle = widgets.ToggleButton(
        value=False,
        description='Show SEM',
        disabled=False,
        button_style='', 
        tooltip='Display Standard Error of Mean',
        layout=widgets.Layout(width='120px')
    )
    
    std_toggle = widgets.ToggleButton(
        value=False,
        description='Show STD',
        disabled=False,
        button_style='', 
        tooltip='Display Standard Deviation',
        layout=widgets.Layout(width='120px')
    )
    
    # Button to trigger the plotting
    plot_button = widgets.Button(
        description='Plot Line',
        disabled=False,
        button_style='', 
        tooltip='Click to plot',
        icon='check'
    )

    def update_diagnosis_options(change):
        new_type = change['new']
        if new_type == 'All Children':
            diagnosis_dropdown.options = ['All Children']
            diagnosis_dropdown.value = 'All Children'
        elif new_type == 'Dyslexia Center Diagnosis':
            unique_vals = extract_dc_diagnoses(df)
            diagnosis_dropdown.options = unique_vals
            diagnosis_dropdown.value = unique_vals[0] if unique_vals else None
        else:
            unique_vals = list(df[new_type].dropna().unique())
            diagnosis_dropdown.options = unique_vals
            diagnosis_dropdown.value = unique_vals[0] if unique_vals else None

    # Observe changes in diagnosis type dropdown and update diagnosis options accordingly
    diagnosis_type_dropdown.observe(update_diagnosis_options, names='value')

    # Function to update the plot
    def update_plot(button):
        # Get the individual's data
        individual_data = df[df[id_col] == int(subject_id)][list(FC_vars.values())].iloc[0]
        
        # Get the comparison group data
        if diagnosis_type_dropdown.value == 'All Children':
            compare_data = df[list(FC_vars.values())]
        elif diagnosis_type_dropdown.value == 'Dyslexia Center Diagnosis':

            compare_subjects = df[df['Dyslexia Center Diagnosis: (choice=' + diagnosis_dropdown.value + ')'] == "Checked"]['ID Number']
            compare_data = df[df['ID Number'].isin(compare_subjects)][list(FC_vars.values())]

        else:
            compare_subjects = df[df[diagnosis_type_dropdown.value] == diagnosis_dropdown.value]['ID Number']
            compare_data = df[df['ID Number'].isin(compare_subjects)][list(FC_vars.values())]

        # Plotting
        fig, ax = plt.subplots(figsize=(24, 12))
        
        # Plot individual data with vivid blue and thicker line
        ax.plot(list(FC_vars.values()), individual_data, marker='o', lw=4, color='blue', label=f"Individual {subject_id}")
        
        # Plot comparison group mean with grey color
        comparison_mean = compare_data.mean()
        comparison_color = 'grey'

        ax.plot(list(FC_vars.values()), comparison_mean, marker='o', lw=3, color=comparison_color, label=f"Mean of {diagnosis_dropdown.value}")
        
        # Adding shaded region for SEM or STD in grey
        if sem_toggle.value:
            sem = compare_data.sem()
            ax.fill_between(list(FC_vars.values()), 
                            comparison_mean - sem, 
                            comparison_mean + sem, 
                            color=comparison_color,
                            alpha=0.2)
        elif std_toggle.value:
            std = compare_data.std()
            ax.fill_between(list(FC_vars.values()), 
                            comparison_mean - std, 
                            comparison_mean + std, 
                            color=comparison_color,
                            alpha=0.2)
        
        # Adjusting aesthetics to match the original function 

        ax.legend(title="Diagnosis", title_fontsize=28, fontsize=24, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title("Individual vs Group Comparison", fontsize=38)
        ax.set_xlabel("Task", fontsize=24)
        ax.set_ylabel("Score", fontsize=24)
        plt.xticks(ticks=range(len(FC_vars)), labels=list(FC_vars.keys()), rotation=90, fontsize=24)
        plt.yticks(ticks=[0, 25, 50, 75, 100], fontsize=24)
        plt.ylim(0, 100)
        # add line at 25, 50, 75
        plt.axhline(y=25, color='grey', linestyle='--', linewidth=1)
        plt.axhline(y=50, color='grey', linestyle='--', linewidth=1)
        plt.axhline(y=75, color='grey', linestyle='--', linewidth=1)
        plt.tight_layout()
    
        # show plot as a python widget
        with out_line:
            out_line.clear_output(wait=True)
            display(fig)

        # Display the interactive table and the initial box plot and kde plot
        # perc_data = df[df['ID Number']==int(subject_id)][list(FC_vars.values())]
        region_selector = create_interactive_table(df, compare_data, int(subject_id), FC_vars, out_plot)
        with out_table:                          
            out_table.clear_output(wait=True)
            display(widgets.HBox([region_selector, out_plot]))

    # Assign the update function to the button
    plot_button.on_click(update_plot)

    # Trigger the initial line plot
    update_plot(None)
    
    # Display the widgets
    # display(Markdown('## Enter an ID number and group of comparison subjects to compare behavioral scores.'))
    display(widgets.HBox([diagnosis_type_dropdown, diagnosis_dropdown, sem_toggle, std_toggle, plot_button]))
    display(widgets.VBox([out_line]))  # Display the Output widget below your other widgets
    display(out_table)  # Display the Output widget below your other widgets


# Define a function that plots the scores for the selected id_number. Use FC_vars to get the labels for the tasks.
# def plot_FC_scores(raw_behavior_df, FC_dict, subject_id, id_col):
#     # Using the keys (domains) and values (lists of column names), plot the scores for each domain on a bar plot
#     fig, ax = plt.subplots(1, 2, figsize=(20, 10))

#     # Plot neuropsych for each list of the scores in each value
#     for domain, tasks in FC_dict.items():
#         # Plot all domains on the same plot, but categorize by domain with domaain labels as well
#         neuropsych_df = raw_behavior_df[tasks]

#     neuropsych_df = raw_behavior_df[FC_dict['neuropsych']]
#     neuropsych_df = neuropsych_df[neuropsych_df[id_col] == subject_id]
#     neuropsych_df = neuropsych_df.T.reset_index()
#     neuropsych_df.columns = ['task', 'score']
#     neuropsych_df['task'] = neuropsych_df['task'].apply(lambda x: x.split(': ')[-1])
#     sns.barplot(x='score', y='task', data=neuropsych_df, ax=ax[0])
#     ax[0].set_title('Neuropsych Scores')
#     ax[0].set_xlabel('Score')
#     ax[0].set_ylabel('Task')
#     ax[0].set_xlim(0, 100)
#     ax[0].set_xticks(range(0, 101, 10))
#     ax[0].set_xticklabels(range(0, 101, 10))
#     ax[0].set_yticklabels(neuropsych_df['task'])
#     ax[0].set_yticks(range(len(neuropsych_df)))
    
#     # Add a dotted red line at 50
#     ax[0].axvline(x=50, color='red', linestyle='--', linewidth=1)

#     return fig

# def plot_FC_scores(raw_behavior_df, FC_dict, subject_id, id_col):
#     """
#     A function to plot FC scores grouped by domain.

#     Parameters:
#         raw_behavior_df (pd.DataFrame): Source data.
#         FC_dict (dict): Mapping of domains to tasks.
#         subject_id (str/int): The subject ID to plot data for.
#         id_col (str): The column name where subject IDs are stored.
    
#     Returns:
#         fig (matplotlib.figure.Figure): The created figure.
#     """
#     all_data = []
#     for domain, tasks in FC_dict.items():
#         for task in tasks:
#             score = raw_behavior_df.loc[raw_behavior_df[id_col] == subject_id, task].values[0]
#             all_data.append([domain, task, score])
#     plot_df = pd.DataFrame(all_data, columns=['domain', 'task', 'score'])

#     colors = ['#0F388A', '#14828C', '#007242', '#443E8C', '#6C247C', '#821A56']

#     # Assign each task a color based on its domain. 
#     # If not enough colors are provided, cycle through the colors.
#     task_colors = {task: colors[i % len(colors)] 
#                 for i, tasks in enumerate(FC_dict.values()) 
#                 for task in tasks}
    
#     plot_df['score'] = pd.to_numeric(plot_df['score'], errors='coerce')
#     plot_df = plot_df.dropna(subset=['score'])

#     fig, ax = plt.subplots(figsize=(18, 10))
#     sns.barplot(x='score', y='task', data=plot_df, palette=task_colors, ax=ax)

#     # Move the plot to the right to make space for the labels
#     ax.set_position([0.4, 0.1, 0.55, 0.8])  # [left, bottom, width, height]

#     # Add space for domain labels
#     # plt.subplots_adjust(left=0.8)  # Adjust left space

#     # Adjust y-ticks and labels to create gaps between domains
#     y_ticks = []
#     y_ticklabels = []
#     start_pos = 0
#     gap = 1  # Adjust as per your requirements
    
#     for domain, tasks in FC_dict.items():
#         end_pos = start_pos + len(tasks) / 2
#         ax.text(-10, (start_pos + end_pos) / 2, domain, 
#                 va='center', ha='right', 
#                 fontweight='bold', fontsize=16)
        
#         y_ticks.extend(list(range(start_pos, start_pos + len(tasks))))
#         y_ticklabels.extend(tasks)
        
#         start_pos += len(tasks) + gap  # Add a gap between domains

#     ax.set_xlim(0, 110)
#     ax.axvline(x=50, color='#E61048', linestyle='--', linewidth=1)
#     ax.set_xlabel('Score')
#     ax.set_ylabel('')

#     return fig

# def plot_FC_scores(raw_behavior_df, FC_dict, subject_id, id_col, title=''):
#     all_data = []
#     for domain, tasks in FC_dict.items():
#         for task in tasks:
#             score = raw_behavior_df.loc[raw_behavior_df[id_col] == subject_id, task].values[0]
#             all_data.append([domain, task, score])
#         # Insert NaN entry to create a gap in the plot
#         all_data.append([domain, None, 0])
#     plot_df = pd.DataFrame(all_data, columns=['domain', 'task', 'score'])

#     colors = ['#178CCB', '#052049']
#     task_colors = {task: colors[i % len(colors)] 
#                    for i, tasks in enumerate(FC_dict.values()) 
#                    for task in tasks}
    
#     plot_df['score'] = pd.to_numeric(plot_df['score'], errors='coerce')
#     plot_df = plot_df.dropna(subset=['score'])

#     fig, ax = plt.subplots(figsize=(14, 10))  

#     sns.barplot(x='score', y='task', data=plot_df, palette=task_colors, ax=ax, zorder=1)

#     # Move the whole barplot to the right to make space for the domain labels
#     ax.set_position([0.4, 0.1, 0.55, 0.8])  # [left, bottom, width, height]

#     # Add domain labels
#     current_task_pos = 0
#     for i, (domain, tasks) in enumerate(FC_dict.items()):
#         domain_label_x = -20  # Adjust this value to ensure the domain label is placed nicely
#         ax.annotate(domain, 
#                     (domain_label_x, current_task_pos), 
#                     textcoords="offset points",
#                     xytext=(-10,10), ha='center',
#                     fontsize=14, fontweight='bold', 
#                     va='center')

#         # Update position for the next domain label
#         current_task_pos += len(tasks) + 1 
    
#     # Style plot
#     # set x-axis limits: 0 to 100 in steps of 25
#     ax.set_xlim(0, 100)
#     ax.set_xticks(range(0, 101, 25))
#     ax.axvline(x=50, color='#878D96', linestyle='--', linewidth=1, zorder=2)
#     ax.axvspan(25, 75, facecolor='#E1E3E5', alpha=0.5, zorder=0)
#     ax.set_xlim(left=-20)  # Adjust left space
#     ax.set_xlabel('Score')
#     ax.set_ylabel('')

#     return fig


# import plotly.graph_objects as go
# import pandas as pd

# def plotly_FC_scores(raw_behavior_df, FC_dict, subject_id, id_col, title=''):
#     all_data = []
#     task_order = []
#     domain_positions = []

#     # Accumulate task data and keep track of domain label positions
#     position_counter = 0
#     for domain, tasks in FC_dict.items():
#         domain_positions.append((domain, position_counter + len(tasks)/2))
#         for task in tasks:
#             score = raw_behavior_df.loc[raw_behavior_df[id_col] == subject_id, task].values[0]
#             all_data.append([domain, task, score])
#             task_order.append(task)
#         position_counter += len(tasks)
        
#     plot_df = pd.DataFrame(all_data, columns=['domain', 'task', 'score'])
    
#     # Color palette
#     blue_palette = ['#052049', '#07407D', '#0961B0', '#178CCB', '#5AA1D6', '#9CBFE2']
#     colors = {domain: blue_palette[i % len(blue_palette)] 
#                     for i, domain in enumerate(FC_dict.keys())}
#     task_colors = {task: colors[domain] 
#                    for domain, tasks in FC_dict.items()
#                    for task in tasks}

#     # Create figure
#     fig = go.Figure()
    
#     # Add bars
#     for task, color in task_colors.items():
#         fig.add_trace(go.Bar(
#             x=plot_df[plot_df['task']==task]['score'],
#             y=plot_df[plot_df['task']==task]['task'],
#             marker_color=color,
#             orientation='h'
#         ))
    
#     # Style adjustments
#     fig.update_layout(
#         barmode='stack',
#         xaxis_title="Score",
#         yaxis_title="Task",
#         xaxis_range=[0, 100],
#         xaxis_dtick=25,
#         showlegend=False,
#         title_text=title,
#         title_x=0.5,
#         title_font_size=20,
#         height=600,
#         margin=dict(l=200)  # Increase left margin for domain labels
#     )

#     # Add domain labels
#     for domain, pos in domain_positions:
#         fig.add_annotation(
#             text='<b>{}</b>'.format(domain),
#             xref="paper", yref="y",
#             x=0, y=pos,
#             showarrow=False,
#             xanchor="right", yanchor="middle",
#             font=dict(size=14)
#         )

#     # Ensure tasks are plotted in the order specified in task_order
#     fig.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': task_order})
    
#     return fig



import plotly.graph_objects as go
import pandas as pd

def plotly_FC_scores(raw_behavior_df, FC_dict, subject_id, id_col, title=''):
    all_data = []
    task_order = []
    domain_labels = []
    
    for domain, tasks in FC_dict.items():
        for task in tasks:
            score = raw_behavior_df.loc[raw_behavior_df[id_col] == subject_id, task].values[0]
            all_data.append([domain, task, score])
            task_order.append(task)
        
        # Compute the position of the domain label
        middle_idx = len(tasks) // 2
        domain_labels.append({"domain": domain, "task": tasks[middle_idx]})
        
    plot_df = pd.DataFrame(all_data, columns=['domain', 'task', 'score'])
    # drop rows that have NaN values in the score column
    plot_df = plot_df.dropna(subset=['score'])
    display(plot_df)
    
    # Creating a color palette
    blue_palette = ['#052049', '#07407D', '#0961B0', '#178CCB', '#5AA1D6', '#9CBFE2']
    colors = {domain: blue_palette[i % len(blue_palette)] 
                    for i, domain in enumerate(FC_dict.keys())}

    # Create figure
    fig = go.Figure()
    
    # Add bars
    for domain in FC_dict.keys():
        # Add domain label above the first task in the domain
        fig.add_annotation(
            text='<b>{}</b>'.format(domain),
            xref="paper", yref="y",
            x=0, y=plot_df[plot_df['domain']==domain].index[0],
            showarrow=False,
            xanchor="left", yanchor="top",
            font=dict(size=14)
        )
        fig.add_trace(go.Bar(
            x=plot_df[plot_df['domain']==domain]['score'],
            y=plot_df[plot_df['domain']==domain]['task'],
            marker_color=colors[domain],
            orientation='h'
        ))
        # Add a space the height of 1 task between domains
        fig.add_trace(go.Bar(
            x=[None],
            y=[None],
            marker_color='white',
            orientation='h'
        ))

 
    # Style adjustments
    fig.update_layout(
        barmode='stack',
        xaxis_title="Score",
        yaxis_title="",
        xaxis_range=[0, 100],
        xaxis_dtick=25,
        yaxis={'categoryorder': 'array', 'categoryarray': task_order},
        showlegend=False,
        title_text=title,
        title_x=0.5,
        title_font_size=20,
        # height=100,
        margin=dict(l=150)  # Adjust left margin to make space for domain labels
    )


    return fig


