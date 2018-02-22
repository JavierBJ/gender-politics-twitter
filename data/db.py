import pymongo
import pandas as pd
from common import text
from time import time

def export_mongodb_slow(tweets, users):
    client = pymongo.MongoClient()  # Default parameters in local
    db = client.sexism
    
    # Insert all tweets (no updates needed, they are always new)
    if tweets is not None:
        result = db.tweets.insert_many(tweets.to_dict('records')) # Records makes a list of dictionaries (one for record)
        print('Number of Tweets inserted is:', len(result.inserted_ids), 'out of', len(tweets))
    
    # Insert new users, update others with newest info if any
    if users is not None:
        users['gender'] = 0
        insertions = 0
        updates = 0
        for _,row in users.iterrows():
            item = row.to_dict()
            
            try:
                existence = db.users.find_one({'id':item['id']})
            except KeyError:    # Needed for 1st case, where no field can be found (collection empty)
                existence = None
            
            if existence is not None:
                if not equal(existence, item):
                    result_del = db.users.delete_one({'id':item['id']})
                    result_add = db.users.insert_one(item)
                    updates += result_del.deleted_count if result_add.acknowledged else 0 # Increment if both results ok
                # Else don't add anything
            else:
                result = db.users.insert_one(item)
                insertions += 1
        print('Number of Users inserted is:', insertions)
        print('Number of Users updated is:', updates)

def export_mongodb(tweets, users):
    client = pymongo.MongoClient()  # Default parameters in local
    db = client.sexism
    
    # Insert all tweets (no updates needed, they are always new)
    if tweets is not None:
        result = db.tweets.insert_many(tweets.to_dict('records')) # Records makes a list of dictionaries (one for record)
        print('Number of Tweets inserted is:', len(result.inserted_ids), 'out of', len(tweets))
    
    # Insert new users, update others with newest info if any
    if users is not None:
        insertions = 0
        updates = 0
        
        existences = db.users.find({'id':{'$in':list(users['id'])}})
        dict_exist = {elem['id']:elem for elem in existences}
        for _, row in users.iterrows():
            item = row.to_dict()
            
            if item['id'] in dict_exist:    # User present in db
                if not equal(item, dict_exist[item['id']]):
                    result_del = db.users.delete_one({'id':item['id']})
                    result_add = db.users.insert_one(item)
                    updates += result_del.deleted_count if result_add.acknowledged else 0 # Increment if both results ok
                # Else don't add anything
            else:
                result = db.users.insert_one(item)
                insertions += 1
        print('Number of Users inserted is:', insertions)
        print('Number of Users updated is:', updates)

def import_tweets_mongodb(query={}, limit=0):
    return import_mongodb('tweets', query, limit)

def import_users_mongodb(query={}, limit=0):
    return import_mongodb('users', query, limit)

def import_mongodb(coll, query={}, limit=0):
    client = pymongo.MongoClient()
    db = client.sexism
    
    cursor = db[coll].find(query).limit(limit)
    df = pd.DataFrame(list(cursor))
    
    print('Imported', len(df), coll)
    return df

def import_tagged_by_gender_tweets_mongodb(tagged_by, limit=0):
    return import_mongodb('tweets', {tagged_by:{'$in':[1,-1]}}, limit)

def import_tagged_by_author_gender_tweets_mongodb(limit=0):
    return import_tagged_by_gender_tweets_mongodb('author_gender', limit)

def import_tagged_by_receiver_gender_tweets_mongodb(limit=0):
    return import_tagged_by_gender_tweets_mongodb('receiver_gender', limit)


def query_id_by_name(name):
    client = pymongo.MongoClient()
    db = client.sexism
    cursor = db['users'].find({'screen_name':name}).limit(1)
    try:
        return cursor[0]['id']
    except IndexError:
        print('Error getting ID from', name)
        
def update_tweets_by_author_gender():
    updates = 0
    client = pymongo.MongoClient()
    db = client.sexism
    tweets_cursor = db['tweets'].find().batch_size(10)
    for tweet in tweets_cursor:
        if tweet['author_gender']==0:
            its_user = tweet['user_id']
            user = db['users'].find_one({'id':its_user})
            gender = user['gender']
            if gender!=0:
                res = db['tweets'].update_many({'user_id':its_user},{'$set':{'author_gender':gender}})
                updates += res.modified_count
    print('Number of Tweets updated is:', updates)
    
def update_tweets_by_receiver_gender():
    updates = 0
    client = pymongo.MongoClient()
    db = client.sexism
    replies_cursor = db['tweets'].find({'msg':'reply'}).batch_size(10)
    for reply in replies_cursor:
        if reply['receiver_gender']==0:
            its_receiver = reply['in_reply_to_user_id']
            receiver = db['users'].find_one({'id':its_receiver})
            if receiver is not None:
                gender = receiver['gender']
                if gender!=0:
                    res = db['tweets'].update_many({'in_reply_to_user_id':its_receiver},{'$set':{'receiver_gender':gender}})
                    updates += res.modified_count
    print('Number of Tweets updated is:', updates)
        
def equal(json1, json2):
    inter = json1.keys() & json2.keys()
    if len(inter)==len(json1) and len(inter)==len(json2):
        for key in inter:
            if json1[key]!=json2[key] and not (json1[key]!=json1[key] and json2[key]!=json2[key]):
                return False    # NaN!=NaN so we specify that case in the condition
        return True
    else:
        return False

def generate_files(path, min_n, max_n):
    l = []
    for i in range(min_n, max_n+1):
        num = str(i)
        if i<10:
            num = '0' + num
        l.append(path+num+'t.csv')
        l.append(path+num+'mr.csv')
    return l

if __name__=='__main__':
    t = time()
    
    tweets_df = text.create_dataset(generate_files('../dump',1,1))
    users_df = text.create_dataset(generate_files('../dumpusers',1,1))
    users_df['gender'] = 0
    export_mongodb(tweets_df, users_df)
    
    print('Total time:', str(time()-t), 's')
    
    '''
    tweets = import_tweets_mongodb(limit=1000)
    for i in range(100):
        print(tweets['full_text'][i])
    
    print()
    users = import_users_mongodb(query={'screen_name': 'agarzon'}, limit=0)
    for i in range(1):
        print(users['name'][i])'''
    
