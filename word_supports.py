from data import mongo
from common import text, feature_extraction
from common import embeddings as emb

print('Loading tweets and extracting supports...')
db = mongo.DB()
tweets = db.import_tweets_mongodb(limit=1000)
tweets = text.preprocess(tweets)
extr = feature_extraction.BinaryBOW(n=1, access_fn=lambda x:x.get_form())
extr = extr.extract(tweets)

print('\tDone.')
print('Finish input with word END.')
while True:
    w = input('Insert word ->')
    if w=='END':
        break
    else:
        print(w, extr.supports[w])

