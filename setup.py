from common import text, mongo
import time


# Setup configuration
name = 'sample'
min_csv = 1
max_csv = 19

# Setup
t = time.time()
db = mongo.DB(db_name=name)

tweets_df = text.create_dataset(db.generate_files('data/csv/dump', min_csv, max_csv))
for col in ['author_gender', 'receiver_gender']:
    tweets_df[col] = 0
for col in ['person_1', 'person_2', 'sexist_1', 'sexist_2', 'hostile_1', 'hostile_2']:
    tweets_df[col] = 9
tweets_df = text.set_language(tweets_df)
print(tweets_df.shape)
users_df = text.create_dataset(db.generate_files('data/csv/dumpusers', min_csv, max_csv))
users_df['gender'] = 0
print(users_df.shape)
db.export_mongodb(tweets_df, users_df)
print('Time in importing users and tweets:', str(time.time()-t), 's')
