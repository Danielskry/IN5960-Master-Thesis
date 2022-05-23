#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module for scraping Chromium tickets based on a set of labels.
The scraping is somewhat "manual": the labels had to be split up into random chunks
in order to bypass Google's bot detecting system (it didn't let me scrape too many).
"""

from pyshadow.main import Shadow
from selenium import webdriver
from time import sleep
import pandas as pd

def main():
    """ Uses labels in directory /labels to scrape issues from Chromium project
    """
    driver = webdriver.Firefox(executable_path=r'geckodriver.exe') # Need driver!
    url = "https://bugs.chromium.org/p/chromium/issues/detail?id="

    iteration = 1

    # Get labels
    df = pd.read_csv(f'labels/labels{iteration}.csv', sep=',')

    #issues = []

    for index, row in df.iterrows():
        issues = []
        issue = str(row['issue_id'])

        driver.get(url + issue)
        shadow = Shadow(driver)

        # Let page load before scraping
        sleep(1)

        # Keep a list of all the details that will 
        # later be concatenated into string
        text = []

        # from Shadow DOM: 
        # Get title
        title = shadow.find_element(".main-text h1")
        text.append(title.text)

        # Get description
        description = shadow.find_element("mr-comment-content")
        text.append(description.text)

        # Find all comments:
        comments = shadow.find_elements("comment-body")

        comment_list = []

        for comment in comments:
            comment_list.append(comment.text)
        
        all_comments = ' '.join(map(str, comment_list))

        text.append(all_comments)

        # Now append tuple to overall list that keeps all rows
        issues.append((text, row['td']))
        issues_df = pd.DataFrame(issues, columns=['text', 'td'])

        issues_df.to_csv(f"fetched/dataset{iteration}.csv", mode='a', header=False, sep=',', encoding='utf-8', index=False)

    # Make dataframe for all the issues
    # issues_df = pd.DataFrame(issues, columns=['text', 'td'])

    # Write to file
    # issues_df.to_csv("dataset.csv", sep=',', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()