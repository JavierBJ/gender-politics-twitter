from common import text
from data import mongo
from gender import assign_gender
import time


# Setup configuration
name = 'sexism'
min_csv = 1
max_csv = 11

# Setup
t = time.time()
db = mongo.DB(db_name=name)

tweets_df = text.create_dataset(db.generate_files('dump', min_csv, max_csv))
tweets_df['in_reply_to_tweet'] = float('nan')
for col in ['author_gender', 'receiver_gender', 'person_1', 'person_2', 'sexist_1', 'sexist_2', 'sentiment_1', 'sentiment_2']:
    tweets_df[col] = 0
users_df = text.create_dataset(db.generate_files('dumpusers', min_csv, max_csv))
users_df['gender'] = 0
db.export_mongodb(tweets_df, users_df)
print('Time in importing users and tweets:', str(time.time()-t), 's')
