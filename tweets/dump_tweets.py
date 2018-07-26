import twitter
import pickle
import time
from datetime import datetime
from common import text
import configparser
import os

def dump(to_dump_tweets, to_dump_mentions, to_dump_replies, num_file, limit, recover_tweets_since_id):
    config = configparser.ConfigParser()
    config.read('config.ini')
    paths_to_accounts = {config['PATH']['PathAut']:config['PATH']['HandleAut'], 
                         config['PATH']['PathCon']:config['PATH']['HandleCon']}
    TRIES = 5
    
    # If num_file==-1 (default), infer it as the maximum pkl already created + 1
    if num_file==-1:
        max_v = 0
        for f in os.listdir(config['PATH']['Pkl']):
            if 'users' not in f and 'log' not in f:    # Check it's a dumpXX... file, to get where the XX is
                try:
                    v = int(f[4:6])
                    max_v = max(max_v, v)
                except ValueError:
                    print('Warning: unexpected file found in data/pkl directory.')
        num_file = max_v + 1
    
    limit = max(limit, 0)   # Prevents from negative limits
    
    # If recover_tweets_since_id==-1, infer it as the recover_tweets_since_id of the log file of the last dump
    if recover_tweets_since_id==-1:
        max_v = 0
        for f in os.listdir(config['PATH']['Csv']):
            if 'dump' in f and 't' in f[6]:
                try:
                    v = int(f[4:6])
                    if v>max_v:
                        max_v = v
                        max_path = f
                except ValueError:
                    print('Warning: unexpected file found in data/csv directory.')
        if max_v>0: # Case where there are already other dumps and were found successfully
            max_f = open(max_path, 'r')
            total = sum([1 for l in max_f])
            for i, line in enumerate(max_f):
                if i==total-2:
                    recover_tweets_since_id = int(line.split(' ')[-1])
        else:       # Case where this is the first dump, so there is no temporal limitation in recovery
            recover_tweets_since_id = 0
    
    # Related to filenames
    if num_file<10:
        num_file = '0'+str(num_file)
    else:
        num_file = str(num_file)
    
    dump_name = 'dump' + num_file
    log_name = 'log' + num_file
    if to_dump_tweets:
        dump_name += 't'
        log_name += 't'
    if to_dump_mentions:
        dump_name += 'm'
        log_name += 'm'
    if to_dump_replies:
        dump_name += 'r'
        log_name += 'r'
    dump_file = config['PATH']['Pkl'] + dump_name
    log_file = config['PATH']['Pkl'] + log_name
        
    flog = open(log_file+'.txt', 'w')
    flog.write('Date: ' + str(datetime.now()) + '\n')
    
    # Load list of accounts and drop missing values
    t1 = time.time()
    accounts = text.retrieve_accounts(paths_to_accounts)
    print('Total accounts:', len(accounts))
    
    
    api = twitter.Api(access_token_key=config['TWITTER']['AccessToken'], access_token_secret=config['TWITTER']['AccessTokenSecret'], consumer_key=config['TWITTER']['ConsumerKey'], consumer_secret=config['TWITTER']['ConsumerSecret'], sleep_on_rate_limit=True, tweet_mode='extended')
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
                for i in range(limit//100):    # Paging solution
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
                for i in range(limit//100):    # Paging solution
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
                for i in range(limit//100):    # Paging solution
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
    
    # Write useful tweets into log
    t = str(time.time() - t1)
    flog.write('Execution time: ' + t + '\n')
    
    # Close files
    fwrite.close()
    flog.close()
    return dump_name
