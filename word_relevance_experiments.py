from data import mongo
from common import text, feature_extraction
from gender import predict_gender

# Parameters
dbname = 'sexism'
lim = 0
wks = None
sw = True
kwf = None
kwr = 5000
n = 1
top_vars = 100
exp_name = 'Receiver Lasso Gender (top 3000 words considered) (removing stopwords)'
path = 'out_relevance/receiver_lasso_gender_rank3000_sw.txt'

# Main code
# ---------
db = mongo.DB(dbname)

# DB to import
#tweets_df = db.import_tagged_by_author_gender_tweets_mongodb(weeks=wks, limit=lim)
tweets_df = db.import_tagged_by_receiver_gender_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_author_gender_political_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_author_gender_individual_tweets_mongodb(weeks=wks, limit=lim)
#tweets_df = db.import_tagged_by_receiver_gender_individual_tweets_mongodb(weeks=wks, limit=lim)

tweets = text.preprocess(tweets_df)

# Target variable
#labels = tweets['author_gender']
labels = tweets['receiver_gender']

# Feature extraction
#ext = feature_extraction.BinaryBOW(n, lambda x: x.get_lemma(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#ext = feature_extraction.BinaryBOW(n, lambda x: x.get_form(), keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
#ext = feature_extraction.BinaryPOS()
ext = feature_extraction.BinaryBOWGender(n, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)

# Relevance
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=ext)
#rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=ext)
#rel = predict_gender.RelevanceByMutualInfo(tweets, labels, extractor=ext)

# Execution
print('Experiment', exp_name, '...')
print('\t', len(rel.features_idx), 'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=top_vars, to=open(path, 'w'))
