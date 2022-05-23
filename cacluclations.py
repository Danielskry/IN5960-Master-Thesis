#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module for calculations.
Includes:
    - Lead times
    - Issue size calculation
    - Lead time for changes
    - Deployment frequency

Module also has functions for fetching the 
specific months from a project and for
Pearson, Spearman and Kendall correlations.
"""

import matplotlib.pyplot as plt
from dateutil.parser import parse
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import numpy as np
import json

def calculate_issue_size(project):
    """ Calculate issue size for project
    Args:
        project (str): path to project
    Returns:
        df_issue_size (dataframe): dataframe with issue sizes
    """

    df = pd.read_csv(project)

    # Calculate the size for each individual PR
    df['size'] = df['additions'] + df['deletions']/2

    # Calculate the average size of each Jira issue based on the individual PR sizes
    df['sum size'] = df.groupby(['key'])['size'].transform('sum')

    # Return dataframe with issue sizes
    df_issue_size = df[['size', 'sum size']]
    return df_issue_size

def get_project(name):
    """ Fetches data from project.
    Additionally, it may be used to clean data.
    Args:
        name (str): name of the project.
    Returns:
        df (dataframe): dataframe of the project
    """
    df = pd.read_csv(f'../data_final/{name}_final.csv')

    """# Set avg lead time as NaN for issues that is not resolved: 
    for index, row in df.iterrows():
        resolved = row['resolved']
            
        # If resolved is NaN
        if isinstance(resolved, float) and math.isnan(resolved):
            df.at[index, 'grouped lead time per change'] = math.nan # set 'avg lead time' to NaN as well
        else:
            pass
    
    # Convert avg lead time from seconds to days
    df['avg lead time'] = df['avg lead time']/86400"""

    # Fix Sakai date issue (or other formating errors here)
    df['created'] = df['created'].str.replace('-0400', '+0000')
    df['created'] = df['created'].str.replace('-0500', '+0000')

    df['created'] = pd.to_datetime(df['created'])
    df['resolved'] = pd.to_datetime(df['resolved'])

    return df

def calculate_lead_times(project):
    """ Calculate lead times.
    N.B. This is not lead time for changes (see down below)
    Args:
        project (str): path to project
    Returns:
        df (dataframe): dataframe with lead times
    """
    
    df = pd.read_csv(project)
        
    lead_times = []

    # Calculate lead time based on 'github_created_at' and 'github_merged_at'
    for index, row in df.iterrows():
        created = row['github_created_at']
        merged = row['github_merged_at']

        if isinstance(merged, float) and math.isnan(merged):
            # if NaN (ergo, not merged) then lead time is also considered NaN
            lead_times.append(math.nan)
        else:
            created_date = parse(created)
            merged_date = parse(merged)

            time_difference = merged_date - created_date
            lead_time = time_difference.seconds + time_difference.days * 86400 # measured in seconds
            lead_times.append(lead_time)

    # Lead time per change measured in seconds (seconds of lead time per change)
    df['lead time'] = lead_times
    df['lead time per change'] = df['lead time']/df['size']

    # If needed to drop rows where 'github_base_ref' is not equal to "master"
    #df = df[df['github_base_ref'].isin(['master'])] 

    # Group based on 'key' and calculate the mean of all the lead times for that 'key'
    df['grouped lead time'] = df['key'].map(df.groupby(['key'])['lead time'].agg(list))
    df['grouped lead time per change'] = df['key'].map(df.groupby(['key'])['lead time per change'].agg(list))

    # dataframe with lead times
    return df

def get_months(df):
    """ Get the months for a project.
    Removes first five and last five months.
    Args:
        df (dataframe): project dataframe
    Returns:
        months (list): list of months in the project as %Y-%m-%d format
    """

    df = df.sort_values(by=['created'])

    earliest_date = df['created'].min()
    oldest_date = df['created'].max()

    months = pd.date_range(earliest_date, oldest_date, freq='MS').strftime("%Y-%m-%d").tolist()

    # Remove the 5 first and last months
    months = months[5:]
    months = months[:len(months) - 5]

    return months

def get_corr_coefficients(x, y):
    """ Get the months for a project.
    Removes first five and last five months.
    Args:
        x (list): first variable for correlation
        y (list): second variable for correlation
    Returns:
        p_coef (int): Pearson's coefficient
        s_coef (int): Spearman's coefficient
        k_coef (int): Kendall's coefficient
    """

    p_coef, pp_val = pearsonr(x, y)
    print(f"Pearson coeff: {p_coef}, and p-value: {pp_val}")

    s_coef, sp_val = spearmanr(x, y)
    print(f'Spearman coeff: {s_coef}, and p-value: {sp_val}')

    k_coef, kp_val = kendalltau(x, y)
    print(f"Kendall's Rank Correlation: {k_coef}, and p-value: {kp_val}")

    return (p_coef, s_coef, k_coef)

def calculate_lead_time_for_changes(project):
    """ Calculates the lead time for changes (per change)
    on a monthly basis.
    Args:
        project (str): path to project
    Returns:
        td_over_time, lead_time_over_time, 
        months_over_time, month_sizes (tuple): Measurements for the TD, months,
        size for each month and lead time for changes on a monthly basis.
    """

    df = get_project(project)
    months = get_months(df)

    td_over_time = []
    lead_time_over_time = []
    months_over_time = []
    month_sizes = []

    for month in months:
        # Convert dates in df to datetime
        df['created'] = pd.to_datetime(df['created'].dt.strftime('%Y-%m-%d %H:%M:%S'))

        month_df = df
        month_df = month_df[['key', 'created', 
                            'resolved', 'sum size', 
                            'td', 'grouped lead time per change']]

        # Earliest date for every month:
        start_date = datetime.strptime(f'{month}', "%Y-%m-%d")

        # Create dataframes based on month as starting point
        # The dataframe will then include all TD from that start point
        month_df = month_df[~(month_df['created'] <= start_date)]

        # End date for created within this year!
        months_year = datetime.strptime(month, '%Y-%m-%d').strftime("%Y")
        month_df = month_df[pd.to_datetime(month_df['created']).dt.strftime('%Y') == months_year]
            
        # End date for created within this month!
        months_month = datetime.strptime(month, '%Y-%m-%d').strftime("%m")
        month_df = month_df[pd.to_datetime(month_df['created']).dt.strftime('%m') == months_month]

        # Remove TD issues that was never resolved in the future
        month_df = month_df[month_df['resolved'].notna()]

        # Now calculate the metrics
        grouped_cells = []
        avg_for_month = 0

        # Convert the grouped PRs lead time per change into a list value
        for cell in month_df['grouped lead time per change']:
            cell = re.sub(r"nan", "0", cell)
            cell = re.sub(r"inf", "0", cell)

            cell_list = eval(cell)


            grouped_cells.extend(cell_list)
            #grouped_cells.append(sum(map(float,cell_list)))
            
        # Now get the mean from that list, ignore divison by zero and if numpy
        # wants to return NaN (can't take mean)
        if grouped_cells:
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_for_month = np.mean(grouped_cells)

        # If is Nan OR inf (invalid values), then set the value = 0
        if math.isnan(avg_for_month) or math.isinf(avg_for_month):
            avg_for_month = 0

        td_size = (month_df['td'] * month_df['sum size']).sum()

        month_df['avg_for_month'] = avg_for_month
        month_df['td_size'] = td_size

        # Append to lists
        td_over_time.append(td_size)
        lead_time_over_time.append(avg_for_month)
        months_over_time.append([str(month)] * len(month_df.index))

        # Correlate for overall size of the project
        month_size = len(month_df.index)
        month_sizes.append(month_size)

    # Return tuple of measurements
    return (td_over_time, lead_time_over_time, 
            months_over_time, month_sizes)

def calculate_deployment_frequency(project):
    """ Calculates the deployment frequency (deployed value)
    on a monthly basis.
    Args:
        project (str): path to project
    Returns:
        td_over_time, deployment_frequency_over_time, 
        months_over_time, month_sizes (tuple): Measurements for the TD, months,
        size for each month and lead time for changes on a monthly basis.
    """

    df = get_project(project)
    months = get_months(df)

    td_over_time = []
    deployment_frequency_over_time = []
    months_over_time = []
    month_sizes = []

    for month in months:
        # Convert dates in df to datetime
        df['resolved'] = pd.to_datetime(df['resolved'].dt.strftime('%Y-%m-%d %H:%M:%S'))

        month_df = df

        # Earliest date for every month:
        #start_date = datetime.strptime(f'{month}-01 00:00:00', "%Y-%m-%d %H:%M:%S")
        start_date = datetime.strptime(f'{month}', "%Y-%m-%d")

        # Create dataframes based on month as starting point
        # The dataframe will then include all TD from that start point
        month_df = month_df[~(month_df['resolved'] <= start_date)]

        # End date for resolved within this year!
        months_year = datetime.strptime(month, '%Y-%m-%d').strftime("%Y")
        month_df = month_df[pd.to_datetime(month_df['resolved']).dt.strftime('%Y') == months_year]
            
        # End date for resolved within this month!
        months_month = datetime.strptime(month, '%Y-%m-%d').strftime("%m")
        month_df = month_df[pd.to_datetime(month_df['resolved']).dt.strftime('%m') == months_month]

        # Remove TD issues that was never resolved in the future
        month_df = month_df[month_df['resolved'].notna()]

        # Now calculate the metrics
        # Average monthly deployment frequency normalized with issue size
        avg_for_month = month_df['sum size'].sum()

        td_size = (month_df['td'] * month_df['sum size']).sum()

        month_df['avg_for_month'] = avg_for_month
        month_df['td_size'] = td_size

        # Append the values to lists
        td_over_time.append(td_size)
        deployment_frequency_over_time.append(avg_for_month)
        months_over_time.append([str(month)] * len(month_df.index))

        # Correlate for overall size of the project
        month_size = len(month_df.index)
        month_sizes.append(month_size)
    
    # Return tuple of measurements
    return (td_over_time, deployment_frequency_over_time, 
            months_over_time, month_sizes)