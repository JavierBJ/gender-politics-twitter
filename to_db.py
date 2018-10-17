from common import mongo, text
import pandas as pd
import time

def safe_div(a,b):
    if b==0:
        return 0
    else:
        return a/b

t0 = time.time()
db = mongo.DB(db_name='sample')
tweets_df = db.import_tagged_by_gender_tweets_mongodb('author_gender')
users_df = db.import_users_mongodb()

output = db.import_tagged_by_author_gender_political_tweets_mongodb()
output = output[['id_str','week','created_at','user_id','author_gender','retweet_count','favorite_count','autname','full_text']]
output = output.merge(users_df[['id','screen_name','name','verified','followers_count','friends_count']], left_on='user_id', right_on='id', how='left')
print('\tJoined columns from Users. (', time.time()-t0, ')')

aux = pd.DataFrame()
aux['reply_count'] = pd.Series([len(tweets_df[tweets_df['in_reply_to_status_id']==x]) for x in output['id_str']])
#print(aux['reply_count'].values[0:200])
print('\tAdded Reply Count. (',time.time()-t0,')')
reply_user_count = pd.Series([len(tweets_df[tweets_df['in_reply_to_user_id']==x]) for x in output['user_id']])
#print(reply_user_count.values[0:200])
aux['reply_normalized'] = pd.Series([safe_div(x,y) for x,y in zip(aux['reply_count'], reply_user_count)])
print('\tAdded Reply Normalized. (',time.time()-t0,')')
aux['reply_norm_male'] = pd.Series([safe_div(len(tweets_df[(tweets_df['in_reply_to_status_id']==x) & (tweets_df['author_gender']==1)]),y) for x,y in zip(output['id_str'], aux['reply_count'])])
print('\tAdded Reply Normalized by Male. (',time.time()-t0,')')
aux['reply_norm_female'] = pd.Series([(1-x) if y>0 else 0 for x,y in zip(aux['reply_norm_male'], aux['reply_count'])])
print('\tAdded Reply Normalized by Female. (',time.time()-t0,')')
rt_user_count = pd.Series([sum(pd.to_numeric(output.loc[output['user_id']==x, 'retweet_count'])) for x in output['user_id']])
aux['rt_normalized'] = pd.Series([safe_div(int(x),y) for x,y in zip(output['retweet_count'], rt_user_count)])
print('\tAdded RT Normalized. (',time.time()-t0,')')
fav_user_count = pd.Series([sum(pd.to_numeric(output.loc[output['user_id']==x, 'favorite_count'])) for x in output['user_id']])
aux['fav_normalized'] = pd.Series([safe_div(int(x),y) for x,y in zip(output['favorite_count'], fav_user_count)])
print('\tAdded Fav Normalized. (',time.time()-t0,')')
acc_to_party = text.retrieve_xls_col_by_account(['data/diputados_autonomicos.csv','data/diputados_congreso.csv'],['twitter account', 'handle'],['partido', 'Partido'])
#print(acc_to_party)
aux['party'] = pd.Series([acc_to_party[x.lower()] for x in output['screen_name']])
print('\tAdded Party. (',time.time()-t0,')')
acc_to_autgov = text.retrieve_xls_col_by_account(['data/diputados_autonomicos.csv','data/diputados_congreso.csv'],['twitter account', 'handle'],['gobierno_auto', 'gobierno_auto'])
aux['aut_gov'] = pd.Series([acc_to_autgov[x.lower()] for x in output['screen_name']])
print('\tAdded Autonomical Government. (',time.time()-t0,')')
acc_to_natgov = text.retrieve_xls_col_by_account(['data/diputados_autonomicos.csv','data/diputados_congreso.csv'],['twitter account', 'handle'],['gobierno_central', 'gobierno_central'])
aux['nat_gov'] = pd.Series([acc_to_natgov[x.lower()] for x in output['screen_name']])
print('\tAdded National Government. (',time.time()-t0,')')
tokenized_tweets, _ = text.preprocess(output, 'author_gender')
aux['feminity'] = pd.Series([text.calculate_feminity(x) for x in tokenized_tweets])
print('\tAdded Feminity. (', time.time()-t0,')')

output = pd.concat([output, aux], axis=1)
output = output[['id_str','user_id','created_at','week','screen_name','name','author_gender','verified','autname','party','aut_gov','nat_gov','followers_count','friends_count','retweet_count','rt_normalized','favorite_count','fav_normalized','reply_count','reply_normalized','reply_norm_male','reply_norm_female','feminity']]
output = output.sort_values(['screen_name','week'])
output = output.rename(index=str, columns={'id_str':'tweet_id','created_at':'date','author_gender':'gender','retweet_count':'rt_count','favorite_count':'fav_count','autname':'autonomy','name':'full_name','friends_count':'following_count'})
out_path = 'db.csv'
output.to_csv(out_path, sep=';', encoding='utf-8')
print('DB created.')
