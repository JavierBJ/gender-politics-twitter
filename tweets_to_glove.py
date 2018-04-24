from common import text
from data import mongo

db = mongo.DB()
tweets = db.import_tweets_mongodb()
ws_tweets = text.tokenize(tweets)

f = open('tweets.txt', 'w')
f.write(ws_tweets)
f.close()
