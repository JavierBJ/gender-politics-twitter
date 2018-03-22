from data import mongo

db = mongo.DB('tiny')
data = db.import_mongodb('tweets', limit=1)
print(data.shape[0])

