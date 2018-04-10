from sexism import detection
from data import mongo
from common import feature_extraction
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Common stuff for all methods (data and parameters)
db = mongo.DB()
dataset = db.import_tagged_by_author_gender_political_tweets_mongodb()
target = 'author_gender'
model = GradientBoostingClassifier()
params = {'max_depth': [2,3,4,5,6], 'learning_rate': [0.001, 0.01, 0.1, 1]}
score = 'roc_auc'

dets = []
word_ranks = [100, 500]

for (i,wr) in enumerate(word_ranks):
    ext = feature_extraction.BinaryBOWGender(keep_words_rank=wr, remove_stopwords=True)
    det = detection.Detector(model, ext, dataset, target, params, score)
    met, conf, cv_all, cv_best = det.metrics()
    dets.append(det)
    print('DETECTOR', str(i+1), '-', str(wr), 'WORD FEATURES')
    print(met)
    print(conf)
    print(cv_best)
    print()

# MacroDetector that sticks to the best
print('MACRO-DETECTOR: BEST CONFIGURATION')
mc = detection.MacroDetector(dets, score)
met, conf, cv_all, cv_best = mc.metrics()
print(met)
print(conf)
print(cv_best)
print()

