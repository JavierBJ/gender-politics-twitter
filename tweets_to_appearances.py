from common import text, mongo
from collections import Counter

db = mongo.DB()
tweets_df = db.import_tweets_mongodb()
tweets = tweets_df['full_text']

tk, _, _, _ = text.setup_freeling()
counts = Counter()
for tweet in tweets:
    tokens = tk.tokenize(tweet)
    counts.update({t.get_form() for t in tokens})

with open('appearances.txt', 'w') as f:
    for (word, count) in counts.most_common():
        f.write(word + ' ' + str(count) + '\n')
