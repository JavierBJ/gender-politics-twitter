import pymongo
import pandas as pd
from common import assign_gender
from tweets import tagging
from collections import defaultdict

class DB():
    
    def __init__(self, db_name='sexism'):
        client = pymongo.MongoClient()  # Default parameters in local
        self.db = client[db_name]
        self.saved = defaultdict(dict)
        self.factors = ['screen_name', 'name', 'verified', 'followers_count', 'friends_count']

    def export_mongodb(self, tweets, users, tags=True):
        print('Exporting tweets to MongoDB...')
        if tweets is not None:
            if tags:
                tweets = tagging.aggregate_tags(tweets, samples=None)
            tweets = tweets[tweets['full_text'].str.count('@')<3] # Remove tweets with 3 or more mentions. It's hard to see who they refer to.
            tweets = tweets[~tweets['full_text'].str.startswith('https://')]  # Remove tweets that start by URL. Usually their just a URL.
            tweets = tweets.reset_index(drop=True)
            tweets = tweets.to_dict('records')
            tweet_to_text = {str(tweet['id_str']):tweet['full_text'] for tweet in tweets}
            replies_text = [tweet_to_text.get(tweet['in_reply_to_status_id'], '') for tweet in tweets]
            for tweet, rep in zip(tweets, replies_text):
                tweet.update({'in_reply_to_text':rep})
            print('\tAdded replies text to tweets.')
        if users is not None:
            users = users.drop_duplicates('id')
            users = assign_gender.tag_politicians_gender(users)
            users = assign_gender.tag_gender_from_politicians(users)
            users = assign_gender.tag_gender_from_r(users, 'data/r-genders.csv', 0.4)
            users = assign_gender.tag_gender_from_gender_api(users, 0.4)
            users = users.to_dict('records')
            print('\tAssigned gender to users.')

        if tweets is not None and users is not None:
            user_to_gender = {user['id']:user['gender'] for user in users}
            author_gender = [user_to_gender.get(tweet['user_id'], 0) for tweet in tweets]
            for tweet, gen in zip(tweets, author_gender):
                tweet.update({'author_gender':gen})
            print('\tAdded author gender to tweets.')

            receiver_gender = [user_to_gender.get(tweet['in_reply_to_user_id'], 0) for tweet in tweets]
            for tweet, gen in zip(tweets, receiver_gender):
                tweet.update({'receiver_gender':gen})
            print('\tAdded receiver gender to tweets.')
        
        if tweets is not None:
            print('\tInserting tweets to DB...')
            result = self.db.tweets.insert_many(tweets)
            print('\tNumber of Tweets inserted is:', len(result.inserted_ids), 'out of', len(tweets))
        if users is not None:
            print('\tInserting users to DB...')
            result = self.db.users.insert_many(users)
            print('\tNumber of Users inserted is:', len(result.inserted_ids), 'out of', len(users))

    def import_users_mongodb(self, query={}, limit=0):
        return self.import_mongodb('users', query, limit)
    
    def import_tweets_mongodb(self, query={}, limit=0):
        return self.import_mongodb('tweets', query, limit)

    def import_mongodb(self, coll, query={}, limit=0):
        cursor = self.db[coll].find(query).limit(limit)
        df = pd.DataFrame(list(cursor))
        
        print('Imported', len(df), coll)
        return df

    def _import_query(self, coll, limit):
        filters = []
        
        return self.import_mongodb(coll, filters, limit)
    
    def import_tweets_for_tagging_mongodb(self, limit=0):
        q = {'$and':[{'msg':'reply'}, {'sexist_1':9}, {'sexist_2':9}, {'hostile_1':9}, {'hostile_2':9},{'in_reply_to_text':{'$ne':''}},{'author_gender':{'$ne':0}},{'receiver_gender':{'$ne':0}}]}
        return self.import_mongodb('tweets', q, limit)

    def import_tagged_by_gender_tweets_mongodb(self, tagged_by, msg_type=None, weeks=None, limit=0):
        filters = []
        filters.append({'lang':{'$in':['es', 'none', 'hr']}})	# Hard-coded line for tries with different languages
        if tagged_by is not None:
            filters.append({tagged_by:{'$in':[1,-1]}})
        if msg_type is not None:
            filters.append({'msg':msg_type})
        if weeks is not None:
            filters.append({'week':{'$in':weeks}})
        return self.import_mongodb('tweets', {'$and':filters}, limit)
    
    def import_tagged_by_author_gender_tweets_mongodb(self, limit=0):
        return self.import_tagged_by_gender_tweets_mongodb('author_gender', limit=limit)
    
    def import_tagged_by_opposing_receiver_gender_tweets_mongodb(self, limit=0):
        return self.import_mongodb('tweets', {'$or':[{'$and':[{'author_gender':1},{'receiver_gender':-1}]}, {'$and':[{'author_gender':-1},{'receiver_gender':1}]}]}, limit=limit)
    
    def import_tagged_by_author_gender_political_tweets_mongodb(self, weeks=None, limit=0):
        return self.import_tagged_by_gender_tweets_mongodb('author_gender', msg_type='tweet', weeks=weeks, limit=limit)
        #return self.import_mongodb('tweets', {'$and':[{'msg':'tweet'},{'author_gender':{'$in':[1,-1]}}]}, limit)

    def import_tagged_by_author_gender_individual_tweets_mongodb(self, weeks=None, limit=0):
        return self.import_tagged_by_gender_tweets_mongodb('author_gender', msg_type='reply', weeks=weeks, limit=limit)
        #return self.import_mongodb('tweets', {'$and':[{'msg':'reply'},{'author_gender':{'$in':[1,-1]}}]}, limit)

    def import_tagged_by_receiver_gender_tweets_mongodb(self, limit=0):
        return self.import_tagged_by_gender_tweets_mongodb('receiver_gender', limit=limit)

    def import_tagged_by_hostile_tweets_mongodb(self, limit=0):
        return self.import_mongodb('tweets', {'is_hostile': {'$in': [1,-1]}}, limit)

    def import_tagged_by_sexist_tweets_mongodb(self, limit=0):
        return self.import_mongodb('tweets', {'is_sexist': {'$in': [1,-1]}}, limit)

    def sample_tweets_mongodb(self, agg_clause):
        cursor = self.db['tweets'].aggregate(agg_clause)
        df = pd.DataFrame(list(cursor))
        return df
    
    def sample_untagged_by_humans_tweets_mongodb(self, tweet_type=['tweet','mention','reply'], limit=100):
        return self.sample_tweets_mongodb([{'$match':{'author_gender':{'$in':[1,-1]}}},
                                      {'$match':{'msg':{'$in':tweet_type}, 'sexist_1':9, 'sexist_2':9, 'hostile_1':9, 'hostile_2':9}},
                                      {'$sample':{'size':limit}}])
    
    
    def sample_untagged_by_humans_candidates_mongodb(self, tweet_type=['tweet','mention','reply'], limit=100):
        # Unused
        return self.sample_tweets_mongodb([{'$match':{'author_gender':{'$in':[1,-1]}}},
                                      {'$match':{'msg':{'$in':tweet_type}, 'sexist_1':9, 'sexist_2':9, 'hostile_1':9, 'hostile_2':9}},
                                      {'$sample':{'size':limit}}])
    
    
    def query_id_by_name(self, name):
        cursor = self.db['users'].find({'screen_name':name}).limit(1)
        try:
            return cursor[0]['id']
        except IndexError:
            print('Error getting ID from', name)
    
    def query_name_by_id(self, ident):
        return self.query_field_by_id('screen_name', ident)

    def query_field_by_id(self, field, ident):
        try:
            return self.saved[ident][field]
        except KeyError:
            cursor = self.db['users'].find({'id':ident}).limit(1)
            try:
                for f in self.factors:
                    self.saved[ident][f] = cursor[0][f]
                    if f==field:
                        result = cursor[0][f]
                return result
            except IndexError:
                print('Error getting field', field, 'from', ident)

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

    def count_replies_by_tweet_id(self, id):
        return self.db['tweets'].find({'msg':'reply', 'in_reply_to_status_id':id, 'author_gender':{'$in':[1,-1]}}).count()

    def count_replies_by_user_id(self, id):
        return self.db['tweets'].find({'msg':'reply', 'in_reply_to_user_id':id, 'author_gender':{'$in':[1,-1]}}).count()

    def count_replies_by_tweet_id_and_gender(self, id, gen):
        return self.db['tweets'].find({'msg':'reply', 'in_reply_to_status_id':id, 'author_gender':gen}).count()

    def count_rts_by_user_id(self, id):
        return sum(int(x['retweet_count']) for x in self.db['tweets'].find({'user_id':id},{'retweet_count':1}))

    def count_favs_by_user_id(self, id):
        return sum(int(x['favorite_count']) for x in self.db['tweets'].find({'user_id':id},{'favorite_count':1}))
