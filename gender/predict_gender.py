from data import db
from common import text, feature_extraction
import numpy as np
from sklearn.linear_model import LogisticRegression


class WordRelevancePredictor():
    
    def __init__(self, phrases, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_lemma())):
        self.phrases = phrases
        self.labels = labels
        self.X = np.array(extractor.extract(phrases).encode(phrases))
        self.features_idx = extractor.features_idx
    
    def compute(self):
        print('Not implemented')
        
    def show(self):
        print('Not implemented')
    
class RelevanceByMutualInfo(WordRelevancePredictor):
    def compute(self):
        # Document counts
        N = len(self.labels)
        count_male = sum([1 for x in self.labels if x==1])
        count_female = sum([1 for x in self.labels if x==-1])
        
        # Term counts
        term_counts = []
        for i in range(self.X.shape[1]):
            term_counts.append(sum(self.X[:,i]))
        
        # Term AND Document counts
        term_male_counts = []
        term_female_counts = []
        for i in range(self.X.shape[1]):
            term_male_counts.append(sum([x for j,x in enumerate(self.X[:,i]) if self.labels[j]==1]))
            term_female_counts.append(sum([x for j,x in enumerate(self.X[:,i]) if self.labels[j]==-1]))
        
        # Mutual information for each term and document pair
        results = {}
        for i in range(self.X.shape[1]):
            results[(i,1)] = np.log(N * term_male_counts[i] / (term_counts[i] * count_male))
            results[(i,-1)] = np.log(N * term_female_counts[i] / (term_counts[i] * count_female))
        self.results = results
        return results
    
    def show(self, top=20):
        males = sorted([(self.features_idx[x],v) for (x,y),v in self.results.items() if y==1], key=lambda x:x[1], reverse=True)[:top]
        females = sorted([(self.features_idx[x],v) for (x,y),v in self.results.items() if y==-1], key=lambda x:x[1], reverse=True)[:top]
        print('Male predictors:')
        print('word : mutual_info')
        for (word, value) in males:
            print(word, ':', value)
        print('\nFemale predictors:')
        print('word : mutual_info')
        for (word, value) in females:
            print(word, ':', value)

class RelevanceByRegression(WordRelevancePredictor):
    def compute(self):
        model = LogisticRegression().fit(self.X, self.labels)
        self.results = model.coef_.flatten()
        return self.results
    
    def show(self, top=20):
        l = [(self.features_idx[x], v) for x,v in enumerate(self.results)]  # Generator?
        males = sorted(l, key=lambda x:x[1], reverse=True)[:top]
        females = sorted(l, key=lambda x:x[1])[:top]
        print('Male predictors:')
        print('word : coef : odds_ratio')
        for (word, value) in males:
            print(word, ':', value, ':', np.exp(value))
        print('\nFemale predictors:')
        print('word : -coef : odds_ratio')
        for (word, value) in females:
            print(word, ':', -value, ':', np.exp(-value))

class RelevanceByLassoRegression(RelevanceByRegression):
    def compute(self):
        model = LogisticRegression(penalty='l1').fit(self.X, self.labels)
        self.results = model.coef_.flatten()
        return self.results

if __name__ == '__main__':
    #tweets_df = db.import_tweets_mongodb({'gender':{'$in':[1,-1]}},limit=100)
    #tweets_df = db.import_tagged_by_author_gender_tweets_mongodb(limit=10000)
    #tweets = text.preprocess(tweets_df)
    #labels = tweets_df['author_gender']
    
    tweets_df = db.import_tagged_by_receiver_gender_tweets_mongodb(limit=10000)
    tweets = text.preprocess(tweets_df)
    labels = tweets_df['receiver_gender']
    
    #rel = RelevanceByMutualInfo(tweets, labels, extractor=feature_extraction.BinaryBOW(feature_extraction.access_features))
    #rel = RelevanceByRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(feature_extraction.access_features))
    rel = RelevanceByLassoRegression(tweets, labels, extractor=feature_extraction.BinaryBOW(feature_extraction.access_features))
    mis = rel.compute()
    rel.show(top=100)

