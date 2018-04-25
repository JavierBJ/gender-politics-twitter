import pickle
import os
import numpy as np
import pandas as pd
import sys


num = 15
code = 'mr'

if len(sys.argv)==3:
    num = int(sys.argv[1])
    code = str(sys.argv[2])

if num<10:
    num = '0' + str(num)
else:
    num = str(num)

filename = 'dump'+num+code
out_tweets = filename
out_users = filename[:4] + 'users' + filename[4:]

def process_text(df1, df2):
    df1 = df1.replace('\n|\r', '', regex=True)
    df2 = df2.replace('\n|\r', '', regex=True)
    return df1, df2

def to_csv(recoveries, status_variables, user_stranger_key, user_variables, replies_variables, are_replies):
    df_tweets = pd.DataFrame()
    df_users = pd.DataFrame()
    if len(recoveries)>0:
        if are_replies:
            variables = status_variables + replies_variables
            df_tweets = pd.DataFrame([[str(getattr(i,j)) for j in variables] for i in recoveries], columns = variables)
        else:
            variables = status_variables
            df_tweets = pd.DataFrame([[str(getattr(i,j)) for j in variables] for i in recoveries], columns = variables)
            for var in replies_variables:
                df_tweets[var] = np.nan
        df_tweets[user_stranger_key] = pd.Series([str(x.user.id) for x in recoveries])
        
        df_users = pd.DataFrame([[str(getattr(i.user,j)) for j in user_variables] for i in recoveries], columns = user_variables)
        df_users = df_users.drop_duplicates()
        
    return df_tweets, df_users

df_statuses = pd.DataFrame()
df_users = pd.DataFrame()
for file in os.listdir('.'):
    if file.startswith(filename) and file.endswith('.pkl'):
        tweets_recovered, mentions_recovered, replies_recovered = pickle.load(open(file, 'rb'))
        present_variables = ['id_str', 'created_at', 'full_text', 'retweet_count', 'favorite_count']
        replies_variables = ['in_reply_to_status_id', 'in_reply_to_user_id']
        user_variables = ['id', 'screen_name', 'name', 'description', 'location', 'verified', 'followers_count', 'friends_count', 'statuses_count', 'favourites_count', 'created_at']
        
        df_tweets, df_users_tweets = to_csv(tweets_recovered, present_variables, 'user_id', user_variables, replies_variables, are_replies=False)
        df_mentions, df_users_mentions = to_csv(mentions_recovered, present_variables, 'user_id', user_variables, replies_variables, are_replies=False)
        df_replies, df_users_replies = to_csv(replies_recovered, present_variables, 'user_id', user_variables, replies_variables, are_replies=True)
                
        print(df_tweets.shape, df_mentions.shape, df_replies.shape)
        df_statuses = pd.concat([df_statuses, df_tweets, df_mentions, df_replies])
        df_users = pd.concat([df_users, df_users_tweets, df_users_mentions, df_users_replies])

df_statuses, df_users = process_text(df_statuses, df_users)
df_statuses.to_csv(out_tweets+'.csv', sep=';', encoding='utf-8')
df_users.to_csv(out_users+'.csv', sep=';', encoding='utf-8')
