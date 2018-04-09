from sexism import detection
from data import mongo
from common import feature_extraction
from sklearn.linear_model import LogisticRegression
import pandas as pd

db = mongo.DB()
dataset = db.import_tagged_by_author_gender_political_tweets_mongodb(limit=1000)
target = 'author_gender'
model = LogisticRegression()
extractor = feature_extraction.BinaryBOWGender(keep_words_rank=50, remove_stopwords=True)
params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
score = 'roc_auc'
det = detection.Detector(model, extractor, dataset, target, params, score)

print(det.describe())
print(det.predict())
met, cv_all, cv_best = det.metrics()
print(met)
#print(pd.DataFrame(cv_all))
print(cv_best)
