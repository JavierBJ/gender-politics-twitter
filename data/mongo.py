import pymongo
import pandas as pd
from common import text
from time import time

class DB():
    
    def __init__(self, db_name='sexism'):
        client = pymongo.MongoClient()  # Default parameters in local
        self.db = client[db_name]

    def export_mongodb_slow(self, tweets, users):
        # Insert all tweets (no updates needed, they are always new)
        if tweets is not None:
            result = self.db.tweets.insert_many(tweets.to_dict('records')) # Records makes a list of dictionaries (one for record)
            print('Number of Tweets inserted is:', len(result.inserted_ids), 'out of', len(tweets))
        
        # Insert new users, update others with newest info if any
        if users is not None:
            users['gender'] = 0
            insertions = 0
            updates = 0
            for _,row in users.iterrows():
                item = row.to_dict()
                
                try:
                    existence = self, self.db.users.find_one({'id':item['id']})
                except KeyError:    # Needed for 1st case, where no field can be found (collection empty)
                    existence = None
                
                if existence is not None:
                    if not self._equal(existence, item):
                        result_del = self.db.users.delete_one({'id':item['id']})
                        result_add = self.db.users.insert_one(item)
                        updates += result_del.deleted_count if result_add.acknowledged else 0 # Increment if both results ok
                    # Else don't add anything
                else:
                    result = self.db.users.insert_one(item)
                    insertions += 1
            print('Number of Users inserted is:', insertions)
            print('Number of Users updated is:', updates)
    
    def export_mongodb(self, tweets, users):
        users = users.drop_duplicates('id')
        # Insert all tweets (no updates needed, they are always new)
        if tweets is not None:
            tweets = tweets.to_dict('records') # Records makes a list of dictionaries (one for record)
            result = self.db.tweets.insert_many(tweets)
            print('Number of Tweets inserted is:', len(result.inserted_ids), 'out of', len(tweets))
            
            replies_updated = 0
            for tweet in tweets:
                if tweet['msg']=='reply':
                    original = self.db.tweets.find_one({'id_str':tweet['in_reply_to_status_id']})
                    if original is not None:
                        result = self.db.tweets.update_one({'id_str': tweet['id_str']},{'$set':{'in_reply_to_tweet':original['full_text']}})
                        replies_updated += result.modified_count
            print('Number of Replies updated with original text is:', replies_updated)
        
        # Insert new users, update others with newest info if any
        if users is not None:
            exist = self.db.users.find({})
            if exist.count()>0: # No
                insertions = 0
                updates = 0
                
                existences = self.db.users.find({'id':{'$in':list(users['id'])}})
                dict_exist = {elem['id']:elem for elem in existences}
                for _, row in users.iterrows():
                    item = row.to_dict()
                    
                    if item['id'] in dict_exist:    # User present in db
                        if not self._equal(item, dict_exist[item['id']]):
                            result_del = self.db.users.delete_one({'id':item['id']})
                            result_add = self.db.users.insert_one(item)
                            updates += result_del.deleted_count if result_add.acknowledged else 0 # Increment if both results ok
                        # Else don't add anything
                    else:
                        result = self.db.users.insert_one(item)
                        insertions += 1
                print('Number of Users inserted is:', insertions)
                print('Number of Users updated is:', updates)
            else:
                result = self.db.users.insert_many(users.to_dict('records'))
                print('Number of Users inserted is:', len(result.inserted_ids))
    
    def import_tweets_mongodb(self, query={}, limit=0):
        return self.import_mongodb('tweets', query, limit)
    
    def import_users_mongodb(self, query={}, limit=0):
        return self.import_mongodb('users', query, limit)
    
    def import_mongodb(self, coll, query={}, limit=0):
        cursor = self.db[coll].find(query).limit(limit)
        df = pd.DataFrame(list(cursor))
        
        print('Imported', len(df), coll)
        return df
    
    def import_tagged_by_gender_tweets_mongodb(self, tagged_by, limit=0):
        return self.import_mongodb('tweets', {tagged_by:{'$in':[1,-1]}}, limit)
    
    def import_tagged_by_author_gender_tweets_mongodb(self, limit=0):
        return self.import_tagged_by_gender_tweets_mongodb('author_gender', limit)
    
    def import_tagged_by_receiver_gender_tweets_mongodb(self, limit=0):
        return self.import_tagged_by_gender_tweets_mongodb('receiver_gender', limit)
    
    def import_tagged_by_opposing_receiver_gender_tweets_mongodb(self, limit=0):
        return self.import_mongodb('tweets', {'$or':[{'$and':[{'author_gender':1},{'receiver_gender':-1}]}, {'$and':[{'author_gender':-1},{'receiver_gender':1}]}]}, limit)
    
    def sample_tweets_mongodb(self, agg_clause):
        cursor = self.db['tweets'].aggregate(agg_clause)
        df = pd.DataFrame(list(cursor))
        return df
    
    def sample_untagged_by_humans_tweets_mongodb(self, tweet_type=['tweet','mention','reply'], limit=100):
        return self.sample_tweets_mongodb([{'$match':{'author_gender':{'$in':[1,-1]}}},
                                      {'$match':{'msg':{'$in':tweet_type}, 'sexist_1':0, 'sexist_2':0, 'sentiment_1':0, 'sentiment_2':0}},
                                      {'$sample':{'size':limit}}])
    
    
    def sample_untagged_by_humans_candidates_mongodb(self, tweet_type=['tweet','mention','reply'], limit=100):
        return self.sample_tweets_mongodb([{'$match':{'author_gender':{'$in':[1,-1]}}},
                                      {'$match':{'msg':{'$in':tweet_type}, 'sexist_1':0, 'sexist_2':0, 'sentiment_1':0, 'sentiment_2':0}},
                                      {'$sample':{'size':limit}}])  # TODO: add appearance of candidate words
    
    
    def query_id_by_name(self, name):
        cursor = self.db['users'].find({'screen_name':name}).limit(1)
        try:
            return cursor[0]['id']
        except IndexError:
            print('Error getting ID from', name)
    
    def update_tweets(self, tweets, fields):
        updates = 0
        tweets = tweets.to_dict('records')
        for tweet in tweets:
            tweet_id = tweet['id_str']
            existing = self.db['tweets'].find_one({'id_str': tweet_id})
            if existing is not None:
                set_fields = {'$set':{x:tweet[x] for x in fields}}
                res = self.db['tweets'].update_one({'id_str': tweet_id}, set_fields)
                updates += res.modified_count
        print('Number of Tweets updated is:', updates)
    
    def update_tweets_by_author_gender(self):
        updates = 0
        tweets_cursor = self.db['tweets'].find().batch_size(10)
        for tweet in tweets_cursor:
            if tweet['author_gender']==0:
                its_user = tweet['user_id']
                user = self.db['users'].find_one({'id':its_user})
                gender = user['gender']
                if gender!=0:
                    res = self.db['tweets'].update_many({'user_id':its_user},{'$set':{'author_gender':gender}})
                    updates += res.modified_count
        print('Number of Tweets updated is:', updates)
        
    def update_tweets_by_receiver_gender(self):
        updates = 0
        replies_cursor = self.db['tweets'].find({'msg':'reply'}).batch_size(10)
        for reply in replies_cursor:
            if reply['receiver_gender']==0:
                its_receiver = reply['in_reply_to_user_id']
                receiver = self.db['users'].find_one({'id':its_receiver})
                if receiver is not None:
                    gender = receiver['gender']
                    if gender!=0:
                        res = self.db['tweets'].update_many({'in_reply_to_user_id':its_receiver},{'$set':{'receiver_gender':gender}})
                        updates += res.modified_count
        print('Number of Tweets updated is:', updates)
    
    def generate_files(self, path, min_n, max_n):
        l = []
        for i in range(min_n, max_n+1):
            num = str(i)
            if i<10:
                num = '0' + num
            l.append(path+num+'t.csv')
            l.append(path+num+'mr.csv')
        return l
    
    def _equal(self, json1, json2):
        inter = json1.keys() & json2.keys()
        if len(inter)==len(json1) and len(inter)==len(json2):
            for key in inter:
                if json1[key]!=json2[key] and not (json1[key]!=json1[key] and json2[key]!=json2[key]):
                    return False    # NaN!=NaN so we specify that case in the condition
            return True
        else:
            return False

if __name__=='__main__':
    t = time()
    '''
    tweets_df = text.create_dataset(generate_files('../dump',9,10))
    tweets_df['author_gender'] = 0
    tweets_df['receiver_gender'] = 0
    users_df = text.create_dataset(generate_files('../dumpusers',9,10))
    users_df['gender'] = 0
    export_mongodb(tweets_df, users_df)
    '''
    print('Total time:', str(time()-t), 's')
    
    '''
    tweets = import_tweets_mongodb(limit=1000)
    for i in range(100):
        print(tweets['full_text'][i])
    
    print()
    users = import_users_mongodb(query={'screen_name': 'agarzon'}, limit=0)
    for i in range(1):
        print(users['name'][i])'''
    
