from common import text
from data import mongo
from gender import assign_gender
import time


# Setup configuration
name = 'sexism'
min_csv = 1
max_csv = 11
flag_assign_gender = True

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

if flag_assign_gender:
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
