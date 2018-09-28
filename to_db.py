import pandas as pd
from common import mongo, text

def safe_div(a,b):
    if b==0:
        return 0
    else:
        return a/b


db = mongo.DB(db_name='sample')
tweets_df = db.import_tagged_by_gender_tweets_mongodb('author_gender', limit=10)
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
output['reply_count'] = pd.Series([db.count_replies_by_tweet_id(x) for x in tweets_df['id_str']])
#output['reply_normalized'] = pd.Series([safe_div(db.count_replies_by_tweet_id(x),db.count_replies_by_user_id(y)) for x,y in zip(tweets_df['id_str'], tweets_df['user_id'])])
#output['reply_norm_male'] = pd.Series([safe_div(db.count_replies_by_tweet_id_and_gender(x,1),db.count_replies_by_tweet_id_and_gender(y,1)+db.count_replies_by_tweet_id_and_gender(y,-1)) for x,y in zip(tweets_df['id_str'], tweets_df['user_id'])])
#output['reply_norm_female'] = pd.Series([safe_div(db.count_replies_by_tweet_id_and_gender(x,-1),db.count_replies_by_tweet_id_and_gender(y,1)+db.count_replies_by_tweet_id_and_gender(y,-1)) for x,y in zip(tweets_df['id_str'], tweets_df['user_id'])])
output['followers_count'] = pd.Series([db.query_field_by_id('followers_count', x) for x in tweets_df['user_id']])
output['following_count'] = pd.Series([db.query_field_by_id('friends_count', x) for x in tweets_df['user_id']])
output['rt_count'] = tweets_df['retweet_count']
output['fav_count'] = tweets_df['favorite_count']
output['rt_normalized'] = pd.Series([safe_div(int(x),db.count_rts_by_user_id(y)) for x,y in zip(tweets_df['retweet_count'], tweets_df['user_id'])])
output['fav_normalized'] = pd.Series([safe_div(int(x),db.count_favs_by_user_id(y)) for x,y in zip(tweets_df['favorite_count'], tweets_df['user_id'])])
output['autonomy'] = tweets_df['autname']
# party
# age
# is_national_governor
# is_autonomic_governor
# feminity_index
output['has_emoji'] = pd.Series([text.has_emoji(x) for x in tweets_df['full_text']])

out_path = 'db.csv'
output.to_csv(out_path, sep=';', encoding='utf-8')
