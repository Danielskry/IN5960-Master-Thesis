# -*- coding: utf-8 -*-
""" Logistic regression classifier

This module uses a logistic regression model as a classification model for the OSS project data.
The dataset used to train and test the model can be found `final_dataset.csv`. The data for the 
OSS projects can be found in folder `project_data`.

"""
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
import logging
import json
import sys

def main():
    """ Logistic regression model
    """
    
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    # Use the final dataset
    dataset_name = 'final_dataset.csv'
    df = pd.read_csv(dataset_name, sep=',', header=None, names=['Text', 'Technical Debt'])

    logging.debug(f'Using dataset {dataset_name} with values:')
    logging.debug(df['Technical Debt'].value_counts())

    # Clean 
    logging.debug(f'Cleaning the dataset!')

    ignore = set(stopwords.words('english'))
    stemmer = WordNetLemmatizer()

    for i, row in df.iterrows():
        words = word_tokenize(row['Text'])
        stemmed = []
        for word in words:
            if word not in ignore:
                stemmed.append(stemmer.lemmatize(word))

        df.at[i, 'Text'] = ' '.join(stemmed)

    # Remove single words (this removes a lot of noise)
    df['Text'] = df['Text'].str.replace(' \w ', '', regex=True)

    # Lowercase all text
    df['Text'] = df['Text'].map(lambda x: x if type(x)!=str else x.lower())

    X_raw = df['Text'].values
    y_raw = df['Technical Debt'].values

    # 80/20 split for training/test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, shuffle=True, random_state = 52, test_size=0.20)

    # tf-idf
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    x_train, x_test, y_train, y_test = X_train_tfidf, X_test_tfidf, y_train, y_test
    
    logging.debug(f'Training and testing the model!')

    # N.B. model parameters found with grid search
    model = LogisticRegression(class_weight='balanced', verbose=1, solver='newton-cg', random_state=5, C=10, penalty='l2') # (solver='newton-cg', class_weight='balanced') gives slightly higher f1-score but more diverse results on unseen data?

    model.fit(x_train, y_train) # train the model

    y_pred = model.predict(x_test) # predict the test data

    # Get the performance scores
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    logging.debug(f'Using the model to classify unseen data from the projects!')

    # Use model on unseen text from the OSS projects data
    with open('projects.json') as project_file:
        projects = json.load(project_file)
    
    # Loop through each project
    for project in projects:
        logging.debug(f"inside {project}:")

        project_df = pd.read_csv(f'projects_data/{project}.csv', sep=',')

        project_df = project_df.rename(columns={'Text': 'text'})
        project_df = project_df.drop_duplicates(subset='key', keep="last")

        predictions = []

        for text in project_df['text']:
            # Transform text with vectorizer, then predict
            unseen = vectorizer.transform([str(text)]).toarray()
            p_unseen = model.predict_proba(unseen)
            
            # Threshold set at P > 75
            predictions.append((p_unseen[:,1]>= 0.75).astype('int')[0])

        stats = f'{project} has {len(predictions)} rows, where {sum(predictions)} ({round((sum(predictions)/len(predictions)*100), 2)}%) gave TD and {len(predictions)-sum(predictions)} ({round((len(predictions)-sum(predictions))/len(predictions)*100, 2)}%) not TD.'
        logging.debug(stats)
        
        # List of predictions into pd column
        td_df = pd.DataFrame({'td': predictions}) 

        # Concatenate the td predictions with projects data horizontally
        td_df.reset_index(drop=True, inplace=True)
        project_df.reset_index(drop=True, inplace=True)
        df_concat = pd.concat([td_df, project_df], axis=1)

        # Write to folder `classifications_data`
        df_concat.to_csv(f'classifications_data/{project}_classifications.csv', sep=',', index=False)

if __name__ == "__main__":
    main()