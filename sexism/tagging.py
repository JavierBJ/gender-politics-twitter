import numpy as np
import pandas as pd
from data import mongo
import random
import os

def export_untagged_sample(size, n_sample, n_copies, verbose=True):
    db = mongo.DB()
    tweets_df = db.import_tweets_for_tagging_mongodb(limit=size)
    
    persons = np.zeros((size, n_copies//2))
    for i in range(size):   # Assign a person to each judgement of each tweet
        persons[i,:] = random.sample(range(1,n_copies+1),n_copies//2)
    
    for i in range(n_copies//2): # Add each person_i column to the dataframe
        tweets_df['person_'+str(i+1)] = persons[:,i].tolist()
    
    for i in range(n_copies//2):   # Initialize judgment_i columns to 0s
        tweets_df['sexist_'+str(i+1)] = 9
        tweets_df['hostile_'+str(i+1)] = 9
    
    # Reorders df so person, sexist and hostile columns are sorted at the end
    cols = tweets_df.columns.tolist()
    to_move = ['receiver_gender', 'author_gender', 'in_reply_to_text', 'full_text']
    to_move += [x for x in cols if x.startswith('person_')]
    to_move += [x for x in cols if x.startswith('sexist_')]
    to_move += [x for x in cols if x.startswith('hostile_')]
    for c in to_move:
        cols += [cols.pop(cols.index(c))] # Move each column name to end of list
    tweets_df = tweets_df[cols] # Reassign columns in new order to df
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
    df = pd.read_csv(path_judgements+'_1.csv', delimiter=';', dtype='str')
    df = df.drop('Unnamed: 0.1', axis=1)
    for i in range(2,n_copies+1):
        other_df = pd.read_csv(path_judgements+'_'+str(i)+'.csv', delimiter=';', dtype='str')
        for j in range(n_copies//2):
            df['sexist_'+str(j+1)] += other_df['sexist_'+str(j+1)]
    db.update_tweets(df, 'sexist_1', 'sexist_2', 'hostile_1', 'hostile_2')

def aggregate_tags(tweets, samples=None):
    if samples is None:
        files = os.listdir('.')
        samples = range(1, 1+max([int(f[7]) for f in files if f.startswith('sample_')]))

    for s in samples:
        sexist1 = pd.Series()
        sexist2 = pd.Series()
        hostile1 = pd.Series()
        hostile2 = pd.Series()
        copies = range(1, 1+max([int(f[9]) for f in files if f.startswith('sample_'+str(s)+'_')]))
        for c in copies:
            df = pd.read_csv('sample_'+str(s)+'_'+str(c)+'.csv', delimiter=';')
            sexist1, sexist2, hostile1, hostile2 = _aggregate_copies(df, sexist1, sexist2, hostile1, hostile2)
        sexist1 = sexist1.replace(0,9).replace(-1,0)
        sexist2 = sexist2.replace(0,9).replace(-1,0)
        hostile1 = hostile1.replace(0,9).replace(-1,0)
        hostile2 = hostile2.replace(0,9).replace(-1,0)
    
        # Only tag those with agreement
        sexist = pd.Series([x if x==y else 9 for x,y in zip(sexist1, sexist2)])
        hostile = pd.Series([x if x==y else 9 for x,y in zip(hostile1, hostile2)])

        # Do updates based on tweet id
        df = pd.read_csv('sample_'+str(s)+'_1.csv', delimiter=';', dtype='str')
        ids = df['id_str']
        p1s = df['person_1']
        p2s = df['person_2']
        for idx, p1, p2, s1, s2, h1, h2, is_s, is_h in zip(ids, p1s, p2s, sexist1, sexist2, hostile1, hostile2, sexist, hostile):
            tweets.loc[tweets['id_str']==idx, 'is_sexist'] = is_s
            tweets.loc[tweets['id_str']==idx, 'is_hostile'] = is_h
            tweets.loc[tweets['id_str']==idx, 'person_1'] = p1
            tweets.loc[tweets['id_str']==idx, 'person_2'] = p2
            tweets.loc[tweets['id_str']==idx, 'sexist_1'] = s1
            tweets.loc[tweets['id_str']==idx, 'sexist_2'] = s2
            tweets.loc[tweets['id_str']==idx, 'hostile_1'] = h1
            tweets.loc[tweets['id_str']==idx, 'hostile_2'] = h2
    return tweets

def _aggregate_copies(df, sexist1, sexist2, hostile1, hostile2):
    s1 = df['sexist_1'].astype('float').replace(0,-1).replace(9,0)
    s2 = df['sexist_2'].astype('float').replace(0,-1).replace(9,0)
    h1 = df['hostile_1'].astype('float').replace(0,-1).replace(9,0)
    h2 = df['hostile_2'].astype('float').replace(0,-1).replace(9,0)
    
    if sexist1.empty:
        sexist1 = s1
        sexist2 = s2
        hostile1 = h1
        hostile2 = h2
    else:
        sexist1 += s1
        sexist2 += s2
        hostile1 += h1
        hostile2 += h2
    return sexist1, sexist2, hostile1, hostile2

if __name__=='__main__':
    export_untagged_sample(1000, 1, 4)
    #import_tagged_sample('sample_1', 4)
