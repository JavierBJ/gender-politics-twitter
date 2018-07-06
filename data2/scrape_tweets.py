import got3 as got
from common import text
import pickle

def scrape(criteria, type_name, tries):
    '''
    criteria = got.manager.TweetCriteria object with the query to scrape
    type_name = String with the type of tweet being downloaded (Tweets, Mentions, Replies)
    tries = Number of attempts to connect with Twitter (prevents from disconnection errors)
    '''
    tweets = []
    for j in range(tries):
        try:
            tweets = got.manager.TweetManager.getTweets(criteria)
            print('\t', type_name, '-', len(tweets))
            break
        except:
            print('\tError retrieving', type_name, 'at attempt', str((j+1)))
    return tweets

retrieved_tweets = []
retrieved_mentions = []
retrieved_replies = []
paths_to_accounts = {'../diputados_autonomicos.csv':'twitter account', '../diputados_congreso.csv':'handle'}
download_tweets = False
download_mentions = True
download_replies = True
retrieval_limit = 100
since_date = '2017-01-01'   # Since and Until when to retrieve tweets, in format 'yyyy-mm-dd'
until_date = '2017-01-31'
from_account_nb = None      # Specify indexes of the list of accounts to start and stop scraping
to_account_nb = None        # Allows to retrieve execution at some point when it gets stuck
tries = 5
dump_file = 'scrape01'


if __name__=='__main__':
    accounts = text.retrieve_accounts(paths_to_accounts)
    print('Accounts available:', len(accounts))
    if from_account_nb is None or from_account_nb<0:
        from_account_nb = 0
    if to_account_nb is None or to_account_nb>len(accounts):
        to_account_nb = len(accounts)
    
    for i, account in enumerate(accounts):
        if i<from_account_nb:
            continue    # Jump until from_account_nb
        if i>to_account_nb:
            break       # Stop after to_account_nb
        print('Account', str(i+1), '-', account)
        
        # Tweets from politicians
        if download_tweets:
            tweet_criteria = got.manager.TweetCriteria().setUsername(account).setSince(since_date).setUntil(until_date).setMaxTweets(retrieval_limit)
            tweets = scrape(tweet_criteria, 'Tweets', tries)
            retrieved_tweets.append(tweets)
            
        # Mentions to politicians
        if download_mentions:
            mention_criteria = got.manager.TweetCriteria().setQuerySearch('@'+account).setSince(since_date).setUntil(until_date).setMaxTweets(retrieval_limit)
            mentions = scrape(mention_criteria, 'Mentions', tries)
            retrieved_mentions.append(mentions)
            
        # Replies to politicians
        if download_replies:
            reply_criteria = got.manager.TweetCriteria().setQuerySearch('to:'+account).setSince(since_date).setUntil(until_date).setMaxTweets(retrieval_limit)
            replies = scrape(reply_criteria, 'Replies', tries)
            retrieved_replies.append(replies)
            
    # Write recoveries to pickle as a tuple (tweets, mentions, replies)
    fwrite = open(dump_file+'.pkl', 'wb')
    pickle.dump((retrieved_tweets, retrieved_mentions, retrieved_replies), fwrite)
