#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import seaborn as sns

from pynteractive.importer import import_dataframe, read_csv_as_list

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


#%%
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



#%%
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

def interactive_individual_line_plot(df, id_col, groupby_col, FC_vars):
    # Dropdown for comparison group selection
    comparison_selector = widgets.Dropdown(
        options=sorted([x for x in df[groupby_col].unique() if isinstance(x, str)]) + ['All Children'],
        value='All Children',
        description='Compare to:',
        disabled=False
    )
    
    # Textbox for user to enter ID
    id_input = widgets.Text(
        value='',
        placeholder='Enter ID Number',
        description='ID Number:',
        disabled=False,
        layout={'width': 'max-content'}
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
        tooltip='Click to plot'
    )
    
    # Create an Output widget
    out = widgets.Output()

    # Function to update the plot
    def update_plot(button):
        # Get the individual's data
        individual_data = df[df[id_col] == int(id_input.value)][list(FC_vars.values())].iloc[0]
        
        # Get the comparison group data
        if comparison_selector.value == 'All Children':
            comparison_data = df[list(FC_vars.values())]
        else:
            comparison_data = df[df[groupby_col] == comparison_selector.value][list(FC_vars.values())]
        
        # Plotting
        fig, ax = plt.subplots(figsize=(24, 12))
        
        # Plot individual data with vivid blue and thicker line
        ax.plot(list(FC_vars.values()), individual_data, marker='o', lw=4, color='blue', label=f"Individual {id_input.value}")
        
        # Plot comparison group mean with grey color
        comparison_mean = comparison_data.mean()
        comparison_color = 'grey'

        ax.plot(list(FC_vars.values()), comparison_mean, marker='o', lw=3, color=comparison_color, label=f"Mean of {comparison_selector.value}")
        
        # Adding shaded region for SEM or STD in grey
        if sem_toggle.value:
            sem = comparison_data.sem()
            ax.fill_between(list(FC_vars.values()), 
                            comparison_mean - sem, 
                            comparison_mean + sem, 
                            color=comparison_color,
                            alpha=0.2)
        elif std_toggle.value:
            std = comparison_data.std()
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
        with out:
            out.clear_output(wait=True)
            display(fig)

    # Assign the update function to the button
    plot_button.on_click(update_plot)
    
    # Display the widgets
    # display(Markdown('## Enter an ID number and group of comparison subjects to compare behavioral scores.'))
    display(widgets.HBox([id_input, comparison_selector, sem_toggle, std_toggle, plot_button]))
    display(widgets.VBox([out]))  # Display the Output widget below your other widgets
