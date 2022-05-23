#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module uses the Github API to fetch data on pull-requests (PRs)
associated with the projects. Then, it stores the JSON response.
"""

import github
import datetime
import json
import os
import re

def append_to_file(file_name, data):
    """ Writes data to a file.

    Args:
        file_name (str): full path to file
        data (str): Data that will be written to the file
    
    """

    # If file does not exist, create one
    if not os.path.isfile(file_name):
        open(file_name, "x")
    
    # Write to file
    f = open(file_name, "a", encoding="utf-8")

    f.write(str(data))
    f.close()

def main():
    """ Fetched data from API requests related to selected projects
    """
    g = github.Github("") # Need to authenticate (e.g., OAuth)

    # For debugging purposes
    # github.enable_console_debug_logging()

    project = "wiredtiger"
    regex_key = "\w{2}-\d*" # Changes for each project. E.g., wiredtiger can be WT-2390, WT-2393, etc.

    repo = g.get_repo(f"{project}/{project}")
    pulls = repo.get_pulls(state='closed') # exclude any open PRs

    # List of filtered pull-requests
    pr_list = []

    for pr in pulls:
        print("inside PR")
        
        merge_date = None
        
        # Make sure that the loop does not exclude pull-request that was NOT merged
        if pr.merged_at is not None:
            merge_date = pr.merged_at.strftime('%A %b %d, %Y at %H:%M GMT')
        
        # Discard all the pull-requests where there is a None on any of these fields
        if not [x for x in (pr.number, pr.state, pr.title, pr.body, pr.base.ref,
                            pr.created_at, pr.updated_at, pr.closed_at
                            ) if x is None]:
            
            # Find key out from the title using regex
            s = re.search(regex_key, pr.title)
            key = s.group(0) if s else None

            # Append a record to the list of pull-requests
            pr_list.append({
                "key": key,
                "number": pr.number,
                "state": pr.state,
                "title": pr.title,
                "body": pr.body,
                "base_ref": pr.base.ref,
                "created_at": pr.created_at.strftime('%A %b %d, %Y at %H:%M GMT'),
                "updated_at": pr.updated_at.strftime('%A %b %d, %Y at %H:%M GMT'),
                "closed_at": pr.closed_at.strftime('%A %b %d, %Y at %H:%M GMT'),
                "merged_at": merge_date,
				"additions": pr.additions,
				"deletions": pr.deletions,
				"changed_files": pr.changed_files,
            })

    # Store the records as a JSON file
    append_to_file(f"{project}_pulls.json", json.dumps(pr_list, indent=4))

if __name__ == '__main__':
    main()