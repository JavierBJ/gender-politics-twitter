import twitter
import csv
import pandas as pd
import pickle

consumer_key = "NxuRecFxAJEHq2gICNb67SuL0"
consumer_secret = "H4yZig0A2oXUXCaBgRLNMZZgIPRCc8ZhK7MrlP6WGNG5qjMP25"
access_token = "184798106-6IC8AWCnwB0UpKk3lWWYNdR0P1UtKOSXFwDAQHmC"
access_token_secret = "jnTp1gDykV7th52e7GXoenRpzCGLOIabTRr0GZKdfCsny"

dataset = pd.read_csv('../diputados_autonomicos.csv', sep=';', encoding='ISO-8859-1')
accounts = dataset['twitter account'].dropna()

#api = twitter.Twitter(auth=twitter.OAuth(access_token, access_token_secret, consumer_key, consumer_secret), retry=10, tweet_mode='extended')
api = twitter.Api(access_token_key=access_token, access_token_secret=access_token_secret, consumer_key=consumer_key, consumer_secret=consumer_secret, sleep_on_rate_limit=True, tweet_mode='extended')

'''print('Showing recoveries for', len(accounts), 'accounts')
for i, account in enumerate(accounts):
    try:
        to = api.search.tweets(q='to:'+accounts[i], count=999999)
        at = api.search.tweets(q='@'+accounts[i], count=999999)
        print(account, len(list(to['statuses'])), len(list(at['statuses'])))
        
        if (i==50):
            break
    except:
        print('Error in', i)'''
        
acc = 'InesArrimadas'

maxid=0
at = []
total=0
for i in range(100000):
    at = api.GetSearch(term='@'+acc+' -filter:retweets', count=999999, max_id=maxid)
    if len(at)>0:
        total += len(at)
        for status in at:
            print(status.full_text)
        print('Iter',i,'Recovered',total)
        maxid = at[-1].id-1
    else:
        break

        

