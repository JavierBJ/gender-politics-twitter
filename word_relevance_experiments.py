import os
import sys
from common import text, feature_extraction, mongo, predict_gender

# Parameters
dbname = 'sexism'
lim = 100000
wks = None
sw = True
kwf = None
kwr = 5000
n = 1
c_regs = [0.001, 0.01, 0.1, 1]
top_vars = 100
exp_name = 'Author SVM Lemma (rank5000 words considered) (removing stopwords)'
path = 'out_relevance/author_svm_lemma_rank5000_sw'

# Main code
# ---------
db = mongo.DB(dbname)

# DB to import
#tweets_df = db.import_tagged_by_author_gender_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_receiver_gender_tweets_mongodb(weeks=wks, limit=lim)
tweets_df = db.import_tagged_by_author_gender_political_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_author_gender_individual_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_receiver_gender_individual_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_hostile_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_sexist_tweets_mongodb(weeks=wks, limit=lim)

tweets = text.preprocess(tweets_df)

# Target variable
labels = tweets['author_gender']
#labels = tweets['receiver_gender']
#labels = tweets['is_hostile']
#labels = tweets['is_sexist']

# Feature extraction
ext = feature_extraction.BinaryBOW(n, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#ext = feature_extraction.BinaryBOW(n, lambda x: x.get_form(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#ext = feature_extraction.BinaryPOS()
#ext = feature_extraction.BinaryBOWGender(n, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)

best = 0
for c_reg in c_regs:
    # Relevance
    rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=ext, c=c_reg)
    #rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=ext, c=c_reg)
    #rel = predict_gender.RelevanceBySupportVectors(tweets, labels, extractor=ext, c=c_reg)

    # Execution
    print('Experiment', exp_name, ', case', c_reg, '...')
    print('\t', len(rel.features_idx), 'features extracted.')
    mis = rel.compute()
    print('\tRelevances computed.')
    os.makedirs(path, exist_ok=True)
    rel.show(top=top_vars, to=open(path+'/'+'c'+str(c_reg).replace('.','_')+'.txt', 'w'))
    print('Train - Validation')
    print('F1:', rel.scores['Train_F1'], '-', rel.scores['Validation_F1'])
    print('Precision:', rel.scores['Train_Precision'], '-', rel.scores['Validation_Precision'])
    print('Recall:', rel.scores['Train_Recall'], '-', rel.scores['Validation_Recall'])
    print('AUC:', rel.scores['Train_AUC'], '-', rel.scores['Validation_AUC'])
    print()
    
    if rel.scores['Validation_AUC']>best:
        best = rel.scores['Validation_AUC']
        best_rel = rel
        best_c = c_reg

print('Best experiment: C =', best_c)
print('Test scores')
print('F1:', best_rel.scores['Test_F1'])
print('Precision:', best_rel.scores['Test_Precision'])
print('Recall:', best_rel.scores['Test_Recall'])
print('AUC:', best_rel.scores['Test_AUC'])

