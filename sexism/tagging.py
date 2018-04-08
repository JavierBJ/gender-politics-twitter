import numpy as np
import pandas as pd
from data import mongo
import random

def export_untagged_sample(size, n_sample, n_copies, verbose=True):
    db = mongo.DB()
    tweets_df = db.import_tweets_for_tagging_mongodb(limit=size)
    
    persons = np.zeros((size, n_copies//2))
    for i in range(size):   # Assign a person to each judgement of each tweet
        persons[i,:] = random.sample(range(1,n_copies+1),n_copies//2)
    
    for i in range(n_copies//2): # Add each person_i column to the dataframe
        tweets_df['person_'+str(i+1)] = persons[:,i].tolist()
    
    for i in range(n_copies//2):   # Initialize judgment_i columns to 0s
        tweets_df['sexist_'+str(i+1)] = 0
        tweets_df['sentiment_'+str(i+1)] = 0
        
    for i in range(n_copies):   # Make n_copies of the csv to be filled (one per person)
        tweets_df.to_csv('sample_'+str(n_sample)+'_'+str(i+1)+'.csv', sep=';', encoding='utf-8')

    if verbose:
        print('Exported', str(n_copies), 'copies of sample', str(n_sample))
        print('\nNumber of assignments to each person:')
        ids, counts = np.unique(persons, return_counts=True)
        dcounts = dict(zip(ids, counts))
        
        for i in range(len(dcounts.keys())):
            print('\tPerson', str(i+1), ':', dcounts[i+1])

def import_tagged_sample(path_judgements, n_copies):
    db = mongo.DB()
    df = pd.read_csv(path_judgements+'_1.csv', delimiter=';')
    df = df.drop('Unnamed: 0.1', axis=1)
    for i in range(2,n_copies+1):
        other_df = pd.read_csv(path_judgements+'_'+str(i)+'.csv', delimiter=';')
        for j in range(n_copies//2):
            df['sexist_'+str(j+1)] += other_df['sexist_'+str(j+1)]
    db.update_tweets(df, 'sexist_1', 'sexist_2', 'sentiment_1', 'sentiment_2')

def aggregated_tag():
    pass

if __name__=='__main__':
    export_untagged_sample(1000, 1, 4)
    #import_tagged_sample('sample_1', 4)
