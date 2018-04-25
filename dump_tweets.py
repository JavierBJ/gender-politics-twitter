import twitter
import pandas as pd
import pickle
import time
from datetime import datetime
from common import metadata, text


# Set these variables to the desired values before execution
num_file = 15
to_dump_tweets = True
to_dump_mentions = False
to_dump_replies = False
recover_tweets_since_id = 983459141904715776
paths_to_accounts = {'diputados_autonomicos.csv':'twitter account', 'diputados_congreso.csv':'handle'}

# Constants related with maximum number of recoveries possible
LIMIT_TWEET_RECOVERIES = 100
LIMIT_MENTION_RECOVERIES = 100
LIMIT_REPLY_RECOVERIES = 100
TRIES = 5

# Related with filenames
if num_file<10:
    num_file = '0'+str(num_file)
else:
    num_file = str(num_file)
dump_file = 'dump' + num_file
log_file = 'log' + num_file

if to_dump_tweets:
    dump_file += 't'
    log_file += 't'
if to_dump_mentions:
    dump_file += 'm'
    log_file += 'm'
if to_dump_replies:
    dump_file += 'r'
    log_file += 'r'
    
flog = open(log_file+'.txt', 'w')
flog.write('Date: ' + str(datetime.now()) + '\n')

# Load list of accounts and drop missing values
t1 = time.time()
accounts = text.retrieve_accounts(paths_to_accounts)
print('Total accounts:', len(accounts))

api = twitter.Api(access_token_key=metadata.access_token, access_token_secret=metadata.access_token_secret, consumer_key=metadata.consumer_key, consumer_secret=metadata.consumer_secret, sleep_on_rate_limit=True, tweet_mode='extended')

tweets_by_id = {}
mentions_by_id = {}
replies_by_id = {}
tweets_recovered = []
mentions_recovered = []
replies_recovered = []
for j, account in enumerate(accounts):

    try:
        if to_dump_tweets:
            # Tweets recovery, limited to a max value per account
            maxid=0
            tweets = []
            total=0
            for i in range(LIMIT_TWEET_RECOVERIES//100):    # Paging solution
                for k in range(TRIES):
                    try:
                        tweets = api.GetUserTimeline(screen_name=account.replace(' ', ''), count=100, max_id=maxid, since_id=recover_tweets_since_id)
                        break
                    except:
                        w = 'FAIL IN TWEETS RECOVERY: ' + account + str(k)
                        print(w)
                        flog.write(w + '\n')
                
                if len(tweets)>0:
                    total += len(tweets)
                    id_aux = tweets[0].user.id
                    for status in tweets:
                        tweets_by_id[status.id_str] = status.full_text
                        tweets_recovered.append(status)
                    maxid = tweets[-1].id-1 # In next page, recover from this id
                else:
                    break   # Exit loop when no more recoveries
            w = 'TWEETS RECOVERY: ' + str(j) + ' - ' + account + ' - ' + str(total)
            print(w)
            flog.write(w + '\n')
        
        if to_dump_mentions:
            # Mentions recovered, limited by number of pages recovered
            maxid=0
            at = []
            total=0
            for i in range(LIMIT_MENTION_RECOVERIES//100):    # Paging solution
                for k in range(TRIES):
                    try:
                        at = api.GetSearch(term='@'+account+' -filter:retweets', count=100, max_id=maxid)
                        break
                    except:
                        w = 'FAIL IN MENTIONS RECOVERY: ' + account + str(k)
                        print(w)
                        flog.write(w + '\n')
                    
                if len(at)>0:
                    total += len(at)
                    for status in at:
                        mentions_by_id[status.id_str] = status.full_text
                        mentions_recovered.append(status)
                    maxid = at[-1].id-1 # In next page, recover from this id
                else:
                    break   # Exit loop when no more recoveries
            w = 'MENTIONS RECOVERY: ' + str(j) + ' - ' + account + ' - ' + str(total)
            print(w)
            flog.write(w + '\n')
        
        if to_dump_replies:
            # Replies recovered, limited by number of pages recovered
            maxid=0
            to = []
            total=0
            for i in range(LIMIT_REPLY_RECOVERIES//100):    # Paging solution
                for k in range(TRIES):
                    try:
                        to = api.GetSearch(term='to:'+account+' -filter:retweets', count=100, max_id=maxid)
                        break
                    except:
                        w = 'FAIL IN REPLIES RECOVERY: ' + account + str(k)
                        print(w)
                        flog.write(w + '\n')
                
                if len(to)>0:
                    total += len(to)
                    for status in to:
                        replies_by_id[status.id_str] = status.full_text
                        replies_recovered.append(status)
                    maxid = to[-1].id-1 # In next page, recover from this id
                else:
                    break   # Exit loop when no more recoveries
            w = 'REPLIES RECOVERY: ' + str(j) + ' - ' + account + ' - ' + str(total)
            print(w)
            flog.write(w + '\n')
    except twitter.error.TwitterError:
        w = 'ERROR: ' + account
        print(w)
        flog.write(w + '\n')


flog.write('Tweets: ' + str(len(tweets_by_id)) + ' ' + str(len(tweets_recovered)) + '\n')
flog.write('Mentions: ' + str(len(mentions_by_id)) + ' ' + str(len(mentions_recovered)) + '\n')
flog.write('Replies: ' + str(len(replies_by_id)) + ' ' + str(len(replies_recovered)) + '\n')
if len(tweets_recovered)>0:
    flog.write('Newest tweet ID recovered: ' + tweets_recovered[0].id_str + '\n')

# Write recoveries to pickle as a tuple (tweets, mentions, replies)
fwrite = open(dump_file+'.pkl', 'wb')
pickle.dump((tweets_recovered, mentions_recovered, replies_recovered), fwrite)

# Write useful data into log
t = str(time.time() - t1)
flog.write('Execution time: ' + t + '\n')

# Close files
fwrite.close()
flog.close()
