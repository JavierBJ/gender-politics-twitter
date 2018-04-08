from data import mongo
from common import text, feature_extraction
from gender import predict_gender

dbname = 'sexism'
lim = 0

# Main code
# ---------
db = mongo.DB(dbname)

'''# Author
tweets_df = db.import_tagged_by_author_gender_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets['author_gender']

print('Experiment Author Lasso POS...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryPOS())
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/author_lasso_pos.txt', 'w'))

print('Experiment Author Lasso...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_lemma(), keep_words_rank=3000, remove_stopwords=True))
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/author_lasso_lemma_rank3000_sw.txt', 'w'))

print('Experiment Author Ridge...')
rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=3000, remove_stopwords=True))
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/author_ridge_form_rank3000_sw.txt', 'w'))

print('Experiment Author MI...')
rel = predict_gender.RelevanceByMutualInfo(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=1000, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/author_mi_form_rank1000_sw.txt','w'))'''

'''# Receiver
tweets_df = db.import_tagged_by_receiver_gender_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets['receiver_gender']

print('Experiment Receiver Lasso POS...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryPOS())
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/receiver_lasso_pos.txt', 'w'))

print('Experiment Receiver Lasso...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_lemma(), keep_words_rank=5000, remove_stopwords=True))
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/receiver_lasso_lemma_rank5000_sw.txt', 'w'))

print('Experiment Receiver Ridge...')
rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=5000, remove_stopwords=True))
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/receiver_ridge_form_rank5000_sw.txt', 'w'))

print('Experiment Receiver MI...')
rel = predict_gender.RelevanceByMutualInfo(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=1000, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/receiver_mi_form_rank1000_sw.txt','w'))'''

'''# Opposite Receiver
tweets_df = db.import_tagged_by_opposing_receiver_gender_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets['receiver_gender']

print('Experiment Opposite Receiver Lasso...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), remove_stopwords=True))
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/op_receiver_lasso_form_sw.txt', 'w'))

print('Experiment Opposite Receiver Ridge...')
rel = predict_gender.RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), remove_stopwords=True))
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/op_receiver_ridge_form_sw.txt', 'w'))

print('Experiment Opposite Receiver MI...')
rel = predict_gender.RelevanceByMutualInfo(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=1000, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/op_receiver_mi_form_rank1000_sw.txt','w'))

print('Experiment Opposite Receiver Lasso POS...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryPOS())
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/op_receiver_lasso_pos.txt', 'w'))'''

'''# Politician
tweets_df = db.import_tagged_by_author_gender_political_tweets_mongodb(weeks=[1,2,3,4,5,6,7,8,11,12], limit=lim)
tweets = text.preprocess(tweets_df, filter_lang=['ca', 'gl'])
labels = tweets['author_gender']

print('Experiment Politician Lasso...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_lemma(), keep_words_rank=3000, remove_stopwords=True))
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/politician_lasso_lemma_rank3000_sw_8m.txt', 'w'))

print('Experiment Politician MI...')
rel = predict_gender.RelevanceByMutualInfo(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=1000, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/politician_mi_form_rank1000_sw.txt','w'))'''

# Individual
tweets_df = db.import_tagged_by_author_gender_individual_tweets_mongodb(limit=lim)
tweets = text.preprocess(tweets_df)
labels = tweets['author_gender']

print('Experiment Individual Lasso...')
rel = predict_gender.RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOWGender(keep_words_rank=3000, remove_stopwords=True))
print('\t',len(rel.features_idx),'features extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/individual_lasso_gender_rank3000_sw.txt', 'w'))

'''print('Experiment Individual MI...')
rel = predict_gender.RelevanceByMutualInfo(tweets, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_form(), keep_words_rank=1000, remove_stopwords=True))
print('\tFeatures extracted.')
mis = rel.compute()
print('\tRelevances computed.')
rel.show(top=100, to=open('out_relevance/individual_mi_form_rank1000_sw.txt','w'))'''
