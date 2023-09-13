import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def sort_tasks_by_fstatistic(df, tasks, categorical_var='diagnosis_dyslexia_phenotype'):
    """
    Sort tasks based on their f-statistic values.

    Parameters:
    - df: DataFrame containing the data
    - tasks: List of task column names in df
    - categorical_var: The column used for grouping data

    Returns:
    - List of tasks sorted by their f-statistic values in descending order
    """
    f_values = {}
    
    for task in tasks:
        # Filter data to exclude rows with NaN values for the current task
        filtered_df = df.dropna(subset=[task])
        
        # Extract unique categories
        categories = filtered_df[categorical_var].unique()
        
        # Extract data for each category
        data_groups = [filtered_df[filtered_df[categorical_var] == category][task] for category in categories]
        
        # Perform one-way ANOVA
        f_val, _ = stats.f_oneway(*data_groups)
        f_values[task] = f_val

    f_statistic = pd.DataFrame(list(f_values.items()), columns=['Task', 'f_statistic'])
    f_statistic.sort_values(by='f_statistic', ascending=False, inplace=True)
    
    return f_statistic['Task'].tolist()

def perform_clustering(df, columns_for_clustering, cluster_range=(2,8), categorical_var='diagnosis_dyslexia_phenotype'):
    """
    Perform KMeans clustering on the provided dataframe for a range of cluster numbers.

    Parameters:
    - df: DataFrame containing the data
    - columns_for_clustering: List of columns to be used for clustering
    - cluster_range: Tuple indicating the starting and ending number of clusters to evaluate
    - categorical_var: The column used for filtering data (currently not used but kept for potential future use)

    Returns:
    - DataFrame with added columns for cluster assignments for each number of clusters
    """
    
    for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
        # Instantiate the clustering model
        model = KMeans(n_clusters=n_clusters)
        
        # Filter data
        filtered_df = df.dropna(axis=0, subset=columns_for_clustering)
        
        # Fit the data to the model
        model.fit(filtered_df[columns_for_clustering])
        
        # Add the cluster labels to the original dataframe, default to NaN
        df[f'cluster_{n_clusters}'] = np.NaN
        df.loc[filtered_df.index, f'cluster_{n_clusters}'] = model.labels_
    
    return df