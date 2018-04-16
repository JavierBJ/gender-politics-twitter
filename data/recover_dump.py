import pandas as pd
import twitter
from common import metadata
import pickle

api = twitter.Api(access_token_key=metadata.access_token, access_token_secret=metadata.access_token_secret, consumer_key=metadata.consumer_key, consumer_secret=metadata.consumer_secret, sleep_on_rate_limit=True, tweet_mode='extended')

path_csv = '../dump03mr.csv'
path_dump = '../dump03mr.pkl'
df = pd.read_csv(path_csv, delimiter=';', dtype='str')

tweets_recovered = []
mentions_recovered = []
replies_recovered = []
for i, (tweet_id, msg_type) in enumerate(zip(df['id_str'], df['msg'])):
    try:
        tw = api.GetStatus(tweet_id)
        if msg_type=='tweet':
            tweets_recovered.append(tw)
            print(i, 'Added tweet with ID', tweet_id)
        elif msg_type=='mention':
            mentions_recovered.append(tw)
            print(i, 'Added tweet with ID', tweet_id)
        elif msg_type=='reply':
            replies_recovered.append(tw)
            print(i, 'Added tweet with ID', tweet_id)
    except Exception:
        print(i, 'Twitter Error in ID:', tweet_id)
        
# Write recoveries to pickle as a tuple (tweets, mentions, replies)
fwrite = open(path_dump, 'wb')
pickle.dump((tweets_recovered, mentions_recovered, replies_recovered), fwrite)