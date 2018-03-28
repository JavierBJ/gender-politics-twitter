from data import mongo
from common import text, feature_extraction
from gender import predict_gender

dbname = 'sexism'
lim = 0

# Main code
# ---------
db = mongo.DB(dbname)

# Author
tweets_df = db.import_tagged_by_author_gender_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets_df['author_gender']

print('Experiment Author Lasso...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/author_lasso_form_rank500_sw.txt', 'w'))

print('Experiment Author Ridge...')
rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/author_ridge_form_rank500_sw.txt', 'w'))

# Receiver
tweets_df = db.import_tagged_by_receiver_gender_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets_df['receiver_gender']

print('Experiment Receiver Lasso...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/receiver_lasso_form_rank500_sw.txt', 'w'))

print('Experiment Receiver Ridge...')
rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/receiver_ridge_form_rank500_sw.txt', 'w'))

# Opposite Receiver
tweets_df = db.import_tagged_by_opposing_receiver_gender_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets_df['receiver_gender']

print('Experiment Opposite Receiver Lasso...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/op_receiver_lasso_form_rank500_sw.txt', 'w'))

print('Experiment Opposite Receiver Ridge...')
rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/op_receiver_ridge_form_rank500_sw.txt', 'w'))
