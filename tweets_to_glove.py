from common import text, mongo

db = mongo.DB()
tweets = db.import_tweets_mongodb()
ws_tweets = text.tokenize(tweets)

f = open('tweets_50.txt', 'w')
f.write(ws_tweets)
f.close()
