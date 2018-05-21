from sexism import detection
from data import mongo
from common import feature_extraction
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Common stuff for all methods (data and parameters)
db = mongo.DB()
dataset = db.import_tagged_by_hostile_tweets_mongodb()
target = 'is_hostile'
model = DecisionTreeClassifier(class_weight='balanced')
params = {'min_samples_split':[2, 32, 256]}
score = 'roc_auc'

for kwr in [100, 500, 1000, 2000, 5000]:
    print('Keep words rank:', kwr)
    ext = feature_extraction.BinaryBOWGender(n=1, keep_words_rank=kwr, remove_stopwords=True)
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

