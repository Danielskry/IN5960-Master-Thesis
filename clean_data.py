#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module that cleans and finalizes all projects data
"""

from os.path import isfile, join
import pandas as pd
import numpy as np
import json
import os
import logging
import sys

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def parse_csv(path, columns=None):
    """ Uses Pandas to parse through a .csv file.
    Args:
        path (str): full path to .csv file
        columns (list): List of subset of columns
    Returns:
        df (dataframe): Pandas dataframe
    """

    df = pd.read_csv(
        path,
        sep = ",", 
        usecols = columns,
        low_memory = False,
        date_parser = lambda col: pd.to_datetime(col, format='%Y-%m-%d %H:%M'),
    )

    return df

def get_comments(file, exp):
    """ Get data from comments.
    Args:
        file (str): full path to .csv file
        exp (list): Regular expression used to filter out comments
    Returns:
        comments (dataframe): Comments for each file
    """

    comments = pd.read_csv(file, date_parser = lambda col: pd.to_datetime(col, format='%Y-%m-%d %H:%M'))

    # Get the comments using a specified regex
    comments = comments.filter(regex=(exp))

    # Rename comments column names as ['comment'0, 'comment1', ...]
    comments = comments.rename(columns={x:f'comment_{y}' for x, y in zip(comments.columns, range(0, len(comments.columns)))})

    # Return dataframe of comments
    return comments

def process_jira(path, columns, comment_regex, renames):
    """ Processing of Jira data.
    Combines the comments into one single text. 
    Renames columns for Jira project to match a homogeneous format
    for columns, so that it may be efficiently used elsewhere.
    Args:
        path (str): path to the Jira project
        columns (list): columns from Jira project that will be kept
        comment_regex (str): regex for comments in Jira project
        renames (dict): dictionary for how columns should be renamed 
    Returns:
        jira_data (dataframe): dataframe of Jira project
    """

    dataframes = []

    # If there are any chunks of data, get all the files
    for file in os.listdir(path):
        if file.endswith(".csv"):
            # Get data from file
            values = parse_csv(f'{path + file}', columns)

            comments = get_comments(f'{path + file}', comment_regex)
            concated_data = pd.concat(
                [values, comments], 
                axis=1, 
                join="inner"
            )

            # Remove HTML tags from the data
            concated_data = concated_data.replace(r'<[^<>]*>', '', regex=True)

            # Append concated_data to list of dataframes
            dataframes.append(concated_data)

        # Ignore files that do not end with .csv    
        else:
            pass
    
    # Convert the list of chunks of dataframes into a single one
    jira_data = pd.concat(dataframes, ignore_index=True)

    # If there are any columns that needs to be renamed for keeping homogeneous column names
    for renaming in renames:
        jira_data = jira_data.rename(columns=renaming)

    # Return a single concated dataframe
    return jira_data

def process_pr(repository):
    """ Get the PRs from a repository
     Args:
        repository (str): path to repository
    Returns:
        prs (dataframe): PRs from repository
    """

    # Get the pull-requests information
    with open(repository) as repo:
        repo_data = json.load(repo)

    # Turn the JSON into a dataframe
    prs = pd.DataFrame(repo_data)
    prs = prs.add_prefix("github_")
    prs = prs.rename(columns={"github_key": "key"})

    # Return pull-requests as a single dataframe
    return prs

def count_stats(path):
    """ Counts stats for a project.
    This is useful information when debugging
    Args:
        path (str): path to the project
    Returns:
        stats (dataframe): Counts for the project at the path
    """

    # List comprehension of all the files in the selected path
    files = [f for f in os.listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]

    logging.debug(f'Found {len(files)} files in path: {path}')

    # List for all the chunks
    file_list = []

    for file in files:
        df = parse_csv(f'{path + file}')
        file_list.append(df)
    
    # Concatenate all the chunks together
    frame = pd.concat(file_list, axis=0, ignore_index=True)

    # Get data about all columns with prefix "github_" (i.e., pull-requests)
    pr_info = frame.filter(regex=("github_.*"))

    # Since Jira data is repeated for each associated pull-request
    # the dataframe has to be filtered for duplicates on the key column
    jira_info = frame.sort_values(by='key', ascending=False)
    jira_info = jira_info.drop_duplicates(subset=['key', 'title'], keep="first")
    jira_info = jira_info[jira_info.columns.drop(list(jira_info.filter(regex='github_.*')))]

    # Now concatenate the Jira and Github data and get the counts
    concated_data = pd.concat([pr_info, jira_info], ignore_index=True, sort=False)
    stats = concated_data.count()

    # Return the count stats of the project
    return stats

def drop_y(df):
    """ Drops duplicated columns from dataframe on prefix `_y`
    """

    to_drop = [y for y in df if y.endswith('_y')]
    df.drop(to_drop, axis=1, inplace=True)

def text_cleaning(project, df):
    """
    
    """

    logging.debug(f"Cleaning the text in the dataframe.")
    
    # Remove redundant key tags from title
    df['title'] = df['title'].str.replace('(BEAM-\w*)|(FLINK-\w*)|(SAK-\w*)|(WFLY-\w*)|(WT-\w*)', '', regex=True)

    # Remove issues where type = Technical Debt (these were used in the training/test set for ML classifiers)
    df = df[~df['type'].isin(['Technical Debt'])]

    # Get all the comments related to the issue
    df['comment'] = df[[col for col in df if col.startswith('comment')]].astype(str).agg(''.join, axis=1)

    # Merge all the Jira and Github text data into a single column
    df['Text'] = df['title'] + ' ' + df['description'] + ' ' + df['comment'] + ' ' + df['github_title'] + ' ' + df['github_body']
    df = df[df['Text'].notna()]

    # Remove redundant key tags from text
    df['Text'] = df['Text'].str.replace('(nan)', ' ', regex=True)
    df['Text'] = df['Text'].str.replace('(BEAM-\w*)|(FLINK-\w*)|(SAK-\w*)|(WFLY-\w*)|(WT-\w*)', '', regex=True)

    # Remove punctuation
    df['Text'] = df['Text'].str.replace('\W', ' ', regex=True)

    logging.debug(f"Removing labels that have been manually marked.")

    # Get labels that I have manually marked
    labels = pd.read_csv(f'raw_data/labels/{project}.csv') # path to manually marked labels
    issues = labels['issue'].tolist()
    issues = [x.strip(' ') for x in issues]

    # Remove rows from the projects that matches with a label
    df = df[~df['key'].isin(issues)]

    logging.debug(f"Done with regex! Starting normalization.")

    # Normalization of text
    ignore = set(stopwords.words('english'))
    stemmer = WordNetLemmatizer()

    for i, row in df.iterrows():
        words = word_tokenize(row['Text'])
        stemmed = []
        for word in words:
            if word not in ignore:
                stemmed.append(stemmer.lemmatize(word))
            
        df.at[i, 'Text'] = ' '.join(stemmed)

    logging.debug(f"Done with normalization!")
        
    df = df.sort_values('key')

    new_df = df[['key', 
                'Text', 
                'type', 
                'resolution', 
                'created', 
                'resolved', 
                'github_number', 
                'github_state', 
                'github_base_ref', 
                'github_created_at', 
                'github_updated_at', 
                'github_closed_at', 
                'github_merged_at']]

    return new_df

def main():
    """ Uses the config.json file to generate
    JSON files for each project based on a .csv
    file and Github URL.
    """

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    with open('projects.json') as project_file:
        projects = json.load(project_file)
    
    # Loop through each project
    for project in projects:
        path = ""
        repo = ""
        comment_regex = ""
        renames = ""
        columns = None

        logging.debug(f'Initializing project {project}!')

        # Declare variables from settings.json file
        for value in projects[project]:
            path = value['path']
            columns = value['columns']
            repo = value['repository']
            comment_regex = value['comment_regex']
            renames = value['column_renames']

        logging.debug(f'Getting Jira data from {path}')
        jira_data = process_jira(path, columns, 
                                comment_regex, renames)

        
        logging.debug(f'Getting Github pull-requests from {repo}!')
        pr_data = process_pr(repo)

        # Merge the pull-requests and Jira data together based on their key
        logging.debug(f'Merging Jira- and Github data together based on their key')
        data = pd.merge(jira_data, 
                        pr_data, 
                        how="outer", 
                        on="key")
        
        # Sort the dataframe
        data = data.sort_values(by='key', ascending=False)
        
        # Keep only rows where value of key is not NaN
        data = data[data['key'].notna()]

        # Keep only rows where value of title is not NaN
        data = data[data['title'].notna()]

        # Keep only rows where value of github_number is non NaN.
        # I.e., all Jira issue have at least one pull-request associated with it
        data = data[data['github_number'].notna()]

        # Apply text cleaning to the data
        cleaned_data = text_cleaning(project, data)

        # Merge PR size data with the PR's github_number
        ngd_df = pd.read_csv(f'../raw_data/Github/pull_request_sizes/{project}.csv')

        # Merge cleaned data with PR size info, drop duplicates columns
        cleaned_data.reset_index(drop=True, inplace=True)
        df = pd.merge(ngd_df, cleaned_data, on=['github_number'], suffixes=('', '_y'))
        drop_y(df)

        # Write data to files
        # uncomment the lines underneath and create folder `cleaned_data`
        # to use the script. This has been commented out to keep this repo clean.
        #logging.debug(f'Writing all data as files to directory {project}_data.csv')
        #df.to_csv(f'cleaned_data/{project}_cleaned.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()