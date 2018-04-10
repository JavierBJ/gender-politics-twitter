from sexism import detection
from data import mongo
from common import feature_extraction
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Common stuff for all methods (data and parameters)
db = mongo.DB()
dataset = db.import_tagged_by_author_gender_political_tweets_mongodb()
target = 'author_gender'
model = LogisticRegression(class_weight='balanced')
params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
score = 'roc_auc'

# Extractor 1
extractor1 = feature_extraction.BinaryBOWGender(keep_words_rank=50, remove_stopwords=True)
det1 = detection.Detector(model, extractor1, dataset, target, params, score)

print('DETECTOR 1: 50 WORD FEATURES')
met, conf, cv_all, cv_best = det1.metrics()
print(met)
print(conf)
print(cv_best)
print()

# Extractor 2
extractor2 = feature_extraction.BinaryBOWGender(keep_words_rank=3000, remove_stopwords=True)
det2 = detection.Detector(model, extractor2, dataset, target, params, score)

print('DETECTOR 2: 3000 WORD FEATURES')
met, conf, cv_all, cv_best = det2.metrics()
print(met)
print(conf)
print(cv_best)
print()

# MacroDetector that sticks to the best
mc = detection.MacroDetector([det1, det2], score)

print('MACRO-DETECTOR: BEST CONFIGURATION')
met, conf, cv_all, cv_best = mc.metrics()
print(met)
print(conf)
print(cv_best)
print()

