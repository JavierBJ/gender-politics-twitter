import pymongo
import pandas as pd
from common import text
from data import db
from gender import assign_gender, predict_gender
import time

t = time.time()
tweets_df = text.create_dataset(db.generate_files('dump',1,7))
tweets_df['author_gender'] = 0
tweets_df['receiver_gender'] = 0
users_df = text.create_dataset(db.generate_files('dumpusers',1,7))
users_df['gender'] = 0
db.export_mongodb(tweets_df, users_df)
print('Time in importing users and tweets:', str(time.time()-t), 's')

users = db.import_users_mongodb()
users = assign_gender.tag_gender_from_r(users, 'r-genders.csv', 0.4)
users = assign_gender.tag_gender_from_gender_api(users, 0.4)
#users = tag_gender_from_genderize_api(users, 0.4)
t1 = time.time()
db.export_mongodb(None, users)
print('Time in assigning gender to users:', str(time.time()-t1), 's')

t2 = time.time()
db.update_tweets_by_author_gender()
print('Time in assigning author gender to tweets:', str(time.time()-t2), 's')

t3 = time.time()
db.update_tweets_by_receiver_gender()
print('Time in assigning receiver gender to tweets:', str(time.time()-t3), 's')
