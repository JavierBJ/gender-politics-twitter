import os
import sys
from data import mongo
from common import text, feature_extraction
from sexism import detection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

# Parameters
dbname = 'sexism'
lim = 0
wks = None
n = 3
folds = 10

# Main code
# ---------
db = mongo.DB(dbname)

# DB to import
#tweets_df = db.import_tagged_by_author_gender_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_receiver_gender_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_author_gender_political_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_author_gender_individual_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_receiver_gender_individual_tweets_mongodb(weeks=wks, limit=lim)
tweets_df = db.import_tagged_by_hostile_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_sexist_tweets_mongodb(weeks=wks, limit=lim)

tweets_df = text.preprocess(tweets_df)

# Target variable
label = 'is_hostile'

ext = feature_extraction.TfIdfChars(n, lambda x: x.get_form(), keep_words_freq=None, keep_words_rank=5000, remove_stopwords=True)

# Models
l1 = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=100)
l2 = LogisticRegression(class_weight='balanced', C=100)
svm = SVC(class_weight='balanced', C=1000000, probability=True)
mlp = MLPClassifier(hidden_layer_sizes=(500,), learning_rate_init=0.001, alpha=0.001)
rf = RandomForestClassifier(n_estimators=50, max_features=0.9, min_samples_leaf=2)
model = VotingClassifier(estimators=[('l1',l1),('l2',l2),('svm',svm),('mlp',mlp),('rf',rf)], voting='soft')
det = detection.Detector(model, ext, tweets_df, label, None, None, folds)
res = det.compute()

det.describe()
print('Train - Validation - Test')
print('F1:', det.scores['Train_F1'], '-', det.scores['Validation_F1'], '-', det.scores['Test_F1'])
print('Precision:', det.scores['Train_Precision'], '-', det.scores['Validation_Precision'], '-', det.scores['Test_Precision'])
print('Recall:', det.scores['Train_Recall'], '-', det.scores['Validation_Recall'], '-', det.scores['Test_Recall'])
print('AUC:', det.scores['Train_AUC'], '-', det.scores['Validation_AUC'], '-', det.scores['Test_AUC'])
print()

print('Confusion matrix:')
print(det.conf)
print()

errors_by_male = []
errors_by_female = []
errors_to_male = []
errors_to_female = []
print('Error analysis:')
for i, text in enumerate(det.text_test):
    l = det.y_test[i]
    lpred = det.te_preds[i]
    a = det.author_genders[i]
    r = det.receiver_genders[i]
    if l!=lpred:
        text = ' '.join([w.get_form() for s in text for w in s.get_words()])
        print(str(l)+';'+str(a)+';'+str(r)+';'+text)
