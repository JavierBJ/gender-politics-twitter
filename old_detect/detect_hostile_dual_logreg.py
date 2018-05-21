from sexism import detection
from data import mongo
from common import feature_extraction
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Common stuff for all methods (data and parameters)
db = mongo.DB()
dataset = db.import_tagged_by_hostile_tweets_mongodb()
target = 'is_hostile'
model = LogisticRegression(class_weight='balanced')
params = {'C': [0.01, 0.1, 1, 10]}
score = 'roc_auc'

for kwr in [100, 500, 1000, 2000, 5000]:
    print('Keep words rank:', kwr)
    ext_tw = feature_extraction.BinaryBOWGender(n=1, keep_words_rank=kwr, remove_stopwords=True)
    ext_reply = feature_extraction.BinaryBOWGender(n=1, keep_words_rank=kwr, remove_stopwords=True, field='in_reply_to_text')
    ext = feature_extraction.FeatureExtractor([ext_tw, ext_reply])
    det = detection.Detector(model, ext, dataset, target, params, score)
    met, conf, cv_all, cv_best = det.metrics()
    print(met)
    print(conf)
    print(cv_best)

# MacroDetector that sticks to the best
'''print('MACRO-DETECTOR: BEST CONFIGURATION')
mc = detection.MacroDetector(dets, score)
met, conf, cv_all, cv_best = mc.metrics()
print(met)
print(conf)
print(cv_best)
print()'''
