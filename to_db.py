import pandas as pd
from common import mongo, text

db = mongo.DB(db_name='sample')
tweets_df = db.import_tagged_by_gender_tweets_mongodb('author_gender', limit=1000)
#tweets = text.preprocess(tweets_df)
#users_df = db.import_users_mongodb()

output = pd.DataFrame()
output['tweet_id'] = tweets_df['id_str']
output['date'] = tweets_df['created_at']
output['week'] = tweets_df['week']
output['user_id'] = tweets_df['user_id']
output['user_screen_name'] = pd.Series([db.query_name_by_id(x) for x in tweets_df['user_id']])
output['user_full_name'] = pd.Series([db.query_field_by_id('name', x) for x in tweets_df['user_id']])
output['gender'] = tweets_df['author_gender']
output['verified'] = pd.Series([db.query_field_by_id('verified', x) for x in tweets_df['user_id']])
# reply_count and normalization
output['followers_count'] = pd.Series([db.query_field_by_id('followers_count', x) for x in tweets_df['user_id']])
output['following_count'] = pd.Series([db.query_field_by_id('friends_count', x) for x in tweets_df['user_id']])
output['rt_count'] = tweets_df['retweet_count']
output['fav_count'] = tweets_df['favorite_count']
# rt/fav normalization
output['autonomy'] = tweets_df['autname']
# party
# age
# is_national_governor
# is_autonomic_governor
# feminity_index
output['has_emoji'] = pd.Series([text.has_emoji(x) for x in tweets_df['full_text']])

out_path = 'db.csv'
output.to_csv(out_path, sep=';', encoding='utf-8')

