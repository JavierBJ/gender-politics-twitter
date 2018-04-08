from data import mongo
from common import text, feature_extraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
import sys


class WordRelevancePredictor():
    
    def __init__(self, phrases, labels, extractor):
        self.phrases = phrases
        self.labels = labels
        self.X = np.array(extractor.extract(phrases).encode(phrases))
        self.features_idx = extractor.features_idx
        self.supports = extractor.supports
    
    def compute(self):
        print('Not implemented')
        
    def show(self):
        print('Not implemented')
    
class RelevanceByMutualInfo(WordRelevancePredictor):
    def compute_old(self):
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
    
    def compute(self):
        male_results = mutual_info_classif(self.X, self.labels)
        female_results = mutual_info_classif(self.X, [-x for x in self.labels])
        results = {}
        for i in range(self.X.shape[1]):
            results[(i,1)] = male_results[i]
            results[(i,-1)] = female_results[i]
        self.results = results
        return results
    
    def retrieve(self, top=20):
        males = sorted([(self.features_idx[x],v) for (x,y),v in self.results.items() if y==1], key=lambda x:x[1], reverse=True)[:top]
        females = sorted([(self.features_idx[x],v) for (x,y),v in self.results.items() if y==-1], key=lambda x:x[1], reverse=True)[:top]
        return males, females
    
    def show(self, top=20, to=sys.stdout):
        males, females = self.retrieve(top)
        print('Male predictors:', file=to)
        print('word : mutual_info', file=to)
        for (word, value) in males:
            print(word, ':', value, file=to)
        print('\nFemale predictors:', file=to)
        print('word : mutual_info', file=to)
        for (word, value) in females:
            print(word, ':', value, file=to)

class RelevanceByRegression(WordRelevancePredictor):
    def __init__(self, phrases, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_lemma()), predname='Ridge (L2) regression'):
        self.predname = predname
        super().__init__(phrases, labels, extractor)

    def compute(self):
        model = LogisticRegression().fit(self.X, self.labels)
        self.females = sum([1 for l in self.labels if l==-1])
        self.males = sum([1 for l in self.labels if l==1])
        self.score = model.score(self.X, self.labels)
        self.results = model.coef_.flatten()
        return self.results
    
    def retrieve(self, top=20):
        l = [(self.features_idx[x], v, self.supports[self.features_idx[x]]) for (x,v) in enumerate(self.results)]  # Generator?
        males = sorted(l, key=lambda x:x[1], reverse=True)[:top]
        females = sorted(l, key=lambda x:x[1])[:top]
        return males, females
    
    def show(self, top=20, to=sys.stdout):
        print(self.predname)
        print('Variables used:', str(len(self.features_idx)))
        print('Samples (total):', str(self.males + self.females))
        print('\tmales:', str(self.males), str(self.males/(self.males+self.females)))
        print('\tfemales:', str(self.females), str(self.females/(self.males+self.females)))
        males, females = self.retrieve(top)
        print('Male predictors:', file=to)
        print('word : coef : odds_ratio', file=to)
        print('Score:', self.score)
        for (word, value, sup) in males:
            print(word, ':', value, ':', np.exp(value), ':', sup, file=to)
        print('\nFemale predictors:', file=to)
        print('word : -coef : odds_ratio', file=to)
        for (word, value, sup) in females:
            print(word, ':', -value, ':', np.exp(-value), ':', sup, file=to)

class RelevanceByLassoRegression(RelevanceByRegression):
    def __init__(self, phrases, labels, extractor=feature_extraction.BinaryBOW(lambda x: x.get_lemma())):
        super().__init__(phrases, labels, extractor, 'Lasso (L1) regression')

    def compute(self):
        model = LogisticRegression(penalty='l1').fit(self.X, self.labels)
        self.females = sum([1 for l in self.labels if l==-1])
        self.males = sum([1 for l in self.labels if l==1])
        self.score = model.score(self.X, self.labels)
        self.results = model.coef_.flatten()
        return self.results

class BootstrapCandidates():
    def __init__(self, positive_words, negative_words, model, extractor, access_fn, tol=1.5):
        self.positives = positive_words
        self.negatives = negative_words
        self.model = model
        self.access_fn = access_fn
        self.extractor = extractor()
    
    def assign(self, tweets):
        labels = []
        for tweet in tweets.full_text:
            # Count positive and negative terms appearing in a tweet
            pos = 0
            neg = 0
            for sent in tweet:
                pos += sum([1 for w in sent if self.access_fn(w) in self.positives])
                neg += sum([1 for w in sent if self.access_fn(w) in self.negatives])
            
            # If there's significantly more words of a certain class, assign the tweet to such class
            if pos/neg>self.tol:
                labels.append(1)
            elif neg/pos>self.tol:
                labels.append(-1)
            else:
                labels.append(0)
        self.assigned_labels = labels
        
        tweets = [tw for i,tw in enumerate(tweets) if labels[i]!=0]
        labels = [l for l in labels if l!=0]
        self.X = self.extractor.extract(tweets).encode(tweets)
        self.model.fit(self.X, labels)
        return self
        
    def predict(self):
        pass

def count_candidate_words(words):
    pass
