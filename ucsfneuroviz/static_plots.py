import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

def plot_avg_scores_by_group(df, task_order, groupby_col,
                            title, xlabel, ylabel, legend_title, y_range=range(0, 101, 10), legend=True,
                            figsize=(27, 15), fontsize=38, save_path=''):
    # plot the mean scores for each task as a line plot
    fig, ax = plt.subplots(figsize=figsize)
    df[task_order+[groupby_col]].groupby(groupby_col).mean().T.plot(kind='line', ax=ax, marker='o')
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize-6)
    ax.set_ylabel(ylabel, fontsize=fontsize-6)
    if legend:
        ax.legend(title=legend_title, title_fontsize=fontsize-10, fontsize=fontsize-10, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        ax.get_legend().remove()
    plt.xticks(ticks=range(len(task_order)), labels=task_order, rotation=90, fontsize=fontsize-10)
    plt.yticks(ticks=y_range, fontsize=fontsize-10) 
    plt.tight_layout()
    plt.grid()
    # add a grid vertically and horizontally
    ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.25)
    ax.grid(axis='x', color='grey', linestyle='-', linewidth=0.25)
    if save_path:
        # save fig as a PDF file
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_all_scores_by_group(df, task_order, groupby_col, group,
                            title, xlabel, ylabel, legend_title, y_range=range(0, 101, 10), legend=True,
                            figsize=(27, 15), fontsize=38, save_path=''):
    # filter the df to only include the group
    df = df[df[groupby_col]==group]
    # plot the mean scores for each task as a line plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # then plot all the scores for each participant as a grey line
    ax.plot(df[task_order].T, color='grey', alpha=0.5)
    ax.plot(df[task_order].mean().T, color='red', alpha=0.5, marker='o', linewidth=5)
 
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize-6)
    ax.set_ylabel(ylabel, fontsize=fontsize-6)
    if legend:
        ax.legend(title=legend_title, title_fontsize=fontsize-10, fontsize=fontsize-10, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        ax.get_legend().remove()
    plt.xticks(ticks=range(len(task_order)), labels=task_order, rotation=90, fontsize=fontsize-10)
    # make y-ticks every 10 points
    plt.yticks(ticks=y_range, fontsize=fontsize-10)
    plt.tight_layout()
    plt.grid()
    # add a grid vertically and horizontally
    ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.25)
    ax.grid(axis='x', color='grey', linestyle='-', linewidth=0.25)
    if save_path:
        # save fig as a PDF file
        plt.savefig(save_path, bbox_inches='tight')
 
def plot_missing_tasks(df, FC_vars, save_path=''):
    # Make a plot showing white squares for missing values and black squares for present values in df[FC_vars]
    fig, ax = plt.subplots(figsize=(60, 30))
    
    # Calculate the number of missing values per column and sort
    column_missing_counts = df[FC_vars].isnull().sum()
    sorted_columns = column_missing_counts.sort_values(ascending=True).index

    # Sort the DataFrame based on columns with most missing values
    df_sorted = df[FC_vars][sorted_columns]

    # Calculate the number of missing values per row and sort the rows
    row_missing_counts = df_sorted.isnull().sum(axis=1)
    sorted_rows = row_missing_counts.sort_values(ascending=True).index

    # Reorder the DataFrame rows based on the sorted row order
    df_sorted = df_sorted.loc[sorted_rows]

    sns.heatmap(df_sorted.isnull().T, ax=ax, cbar=False, cmap=['black', 'white'], yticklabels=True)
    ax.set_xticklabels([])
    ax.set_yticklabels(df_sorted.columns, fontsize=30)
    if save_path != '':
        plt.savefig(save_path, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_age_stacked_by_group(df, age_col, group_col,
                              title='Age of Dyslexia Center Participants',
                              save_path=''):
    # plot a bar plot for age (rounded to the nearest year) where each bar is an age, and each bar is stacked by diagnosis (dyslexic, control)
    fig, ax = plt.subplots(figsize=(10, 6))
    # round age to the nearest year, ignoring NaNs
    df['age_rounded'] = df[age_col].round()
    # Create a pivot table with counts of participants by age and diagnosis
    age_diagnosis_counts = df.pivot_table(index='age_rounded', columns=group_col, aggfunc='size', fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the stacked bar chart using ax.bar directly
    bottom = None
    for column in age_diagnosis_counts.columns:
        ax.bar(age_diagnosis_counts.index, age_diagnosis_counts[column], label=column, bottom=bottom)
        if bottom is None:
            bottom = age_diagnosis_counts[column]
        else:
            bottom += age_diagnosis_counts[column]

    ax.set_title(title, fontsize=24)
    ax.set_xlabel('Age', fontsize=18)
    ax.set_ylabel('Number of Participants', fontsize=16)

    ax.set_xticks(age_diagnosis_counts.index)
    ax.set_xticklabels(age_diagnosis_counts.index.astype(int), fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(title='Diagnosis', fontsize=14, title_fontsize=14)
    ax.tick_params(axis='x')
    plt.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    if save_path != '':
        plt.savefig(save_path, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_counts_per_group(df, group_col,
                          title='Counts of Dyslexia Phenotypes', x_label='Dyslexia Phenotype',
                          save_path=''):
    # bar plot the counts of each diagnosis
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot the counts of each diagnosis using matplotlib
    plt.bar(df[group_col].value_counts().index, df[group_col].value_counts().values)

    ax.set_title(title, fontsize=24)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel('Number of Participants', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # rotate the x-tick labels and align them to the left
    plt.xticks(rotation=45, ha='right')
    # add y-axis grid lines
    plt.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
    # put counts above each bar
    for i in range(len(df[group_col].value_counts().index)):
        plt.text(i, df[group_col].value_counts().values[i]+1, df[group_col].value_counts().values[i], horizontalalignment='center', fontsize=12)
    if save_path != '':
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_elbow(df, columns_for_clustering, cluster_range=(1,12)):
    """
    Plot the elbow curve to determine the optimal number of clusters.

    Parameters:
    - df: DataFrame containing the data
    - columns_for_clustering: List of columns to be used for clustering
    - cluster_range: Tuple indicating the starting and ending number of clusters to evaluate

    Returns:
    - Optimal number of clusters based on the elbow method
    """
    # Filter data
    data = df[columns_for_clustering].dropna(axis=0)
    
    # Instantiate the KMeans model
    model = KMeans()
    
    # Create an elbow visualizer
    visualizer = KElbowVisualizer(model, k=cluster_range)
    
    # Fit the data to the visualizer and display the plot
    visualizer.fit(data)
    visualizer.show()
    
    return visualizer.elbow_value_
