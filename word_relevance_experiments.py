from data import mongo
from common import text, feature_extraction
from gender import predict_gender

dbname = 'tiny'
lim = 0

# Main code
# ---------
db = mongo.DB(dbname)

# Author
tweets_df = db.import_tagged_by_author_gender_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets_df['author_gender']

rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
mis = rel.compute()
rel.show(top=100, to=open('out_relevance/author_lasso_form_rank500_sw.txt', 'w'))

rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
mis = rel.compute()
rel.show(top=100, to=open('out_relevance/author_ridge_form_rank500_sw.txt', 'w'))

# Receiver
tweets_df = db.import_tagged_by_receiver_gender_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets_df['receiver_gender']

rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
mis = rel.compute()
rel.show(top=100, to=open('out_relevance/receiver_lasso_form_rank500_sw.txt', 'w'))

rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
mis = rel.compute()
rel.show(top=100, to=open('out_relevance/receiver_ridge_form_rank500_sw.txt', 'w'))

# Opposite Receiver
tweets_df = db.import_tagged_by_opposing_receiver_gender_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets_df['receiver_gender']

rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
mis = rel.compute()
rel.show(top=100, to=open('out_relevance/op_receiver_lasso_form_rank500_sw.txt', 'w'))

rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=500, remove_stopwords=True))
mis = rel.compute()
rel.show(top=100, to=open('out_relevance/op_receiver_ridge_form_rank500_sw.txt', 'w'))
