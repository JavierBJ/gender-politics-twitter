import os
import sys
from data import mongo
from common import text, feature_extraction
from sexism import detection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Parameters
dbname = 'sexism'
lim = 0
wks = None
sw = True
kwf = None
n = 3
#n2 = 2
folds = 10
kwrs = [5000]
c_regs = [1000000, 10000000, 100000000]
exts = ['tfidfchars']

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
#label = 'is_sexist'

best = 0
for extname in exts:
    for kwr in kwrs:
        for c_reg in c_regs:
            # Feature extraction
            if extname=='binary':
                ext = feature_extraction.BinaryBOW(n, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#                ext = feature_extraction.BinaryBOWGender(n, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
            elif extname=='counts':
                ext = feature_extraction.CountsBOW(n, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#                ext = feature_extraction.CountsBOWGender(n, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
            elif extname=='tfidf':
                ext = feature_extraction.TfIdfBOW(n, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#                ext = feature_extraction.TfIdfBOWGender(n, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
            elif extname=='binarychars':
                ext = feature_extraction.BinaryChars(n, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
            elif extname=='countschars':
                ext = feature_extraction.CountsChars(n, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
            elif extname=='tfidfchars':
                ext = feature_extraction.TfIdfChars(n, lambda x: x.get_form(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)

#            if extname=='binary':
#                ext2 = feature_extraction.BinaryBOW(n2, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#            elif extname=='counts':
#                ext2 = feature_extraction.CountsBOW(n2, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#            elif extname=='tfidf':
#                ext2 = feature_extraction.TfIdfBOW(n2, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#            ext = feature_extraction.FeatureExtractor([ext, ext2])

            # Model
#            model = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=c_reg)
            model = SVC(class_weight='balanced', C=c_reg)
            det = detection.Detector(model, ext, tweets_df, label, None, None, folds)
            res = det.compute()
            
            print('Experiment: Extractor =', extname, 'KeepWordsRank =', kwr ,'C =', c_reg, '...')
            det.describe()
            print('Train - Validation')
            print('F1:', det.scores['Train_F1'], '-', det.scores['Validation_F1'])
            print('Precision:', det.scores['Train_Precision'], '-', det.scores['Validation_Precision'])
            print('Recall:', det.scores['Train_Recall'], '-', det.scores['Validation_Recall'])
            print('AUC:', det.scores['Train_AUC'], '-', det.scores['Validation_AUC'])
            print()

            if det.scores['Validation_F1']>best:
                best = det.scores['Validation_F1']
                best_det = det
                best_c = c_reg
                best_kwr = kwr
                best_ext = extname

print('Best experiment: Extractor =', best_ext, 'KeepWordsRank =', best_kwr ,'C =', best_c)
print('Test scores')
print('F1:', best_det.scores['Test_F1'])
print('Precision:', best_det.scores['Test_Precision'])
print('Recall:', best_det.scores['Test_Recall'])
print('AUC:', best_det.scores['Test_AUC'])

