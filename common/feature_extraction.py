import pandas as pd
import numpy as np
from data import db
from common import text

class FeatureExtractor():
    def __init__(self):
        print('Not implemented')
        
    def extract(self, source):
        print('Not implemented')
        
    def encode(self, df):
        print('Not implemented')

class FeatureExtractorBOW(FeatureExtractor):
    def __init__(self, inc, top, access_fn):
        if top<1:
            top = float('inf')  # No top limit if not positive
        self.inc = inc
        self.top = top
        self.access_fn = access_fn
        self.features = None
    
    def extract(self, source):
        words = set()
        for tweet in source.full_text:
            for sent in tweet:
                words.update({self.access_fn(x) for x in sent})
        
        self.features = {x:i for i,x in enumerate(words)}
        self.features_idx = list(words)
        
        return self
    
    def encode(self, tweets):
        if self.features is not None:
            encodings = []
            for tweet in tweets.full_text:
                encoding = np.zeros((len(self.features),))  # Prepare encoding vector
                for sent in tweet:
                    for word in sent:
                        w = self.access_fn(word)
                        # Only increment counter if not already in the top
                        if w in self.features and encoding[self.features[w]]<self.top:
                            encoding[self.features[w]] += self.inc
                encodings.append(encoding)
        else:
            raise AttributeError('Features not extracted. Use FeatureExtractor.extract(source) first')
        return encodings

class BinaryBOW(FeatureExtractorBOW):
    def __init__(self, access_fn):
        super().__init__(1,1, access_fn)

class CountsBOW(FeatureExtractorBOW):
    def __init__(self, access_fn):
        super().__init__(1,-1, access_fn)

class Embedding(FeatureExtractor):
    def extract(self, pre_trained_emb):
        self.embeddings = pre_trained_emb
        self.dim = len(next(iter(pre_trained_emb.values())))
        return self
    
    def encode(self, tweets):
        print('Not implemented')    # TODO: do subclasses for word embeddings by aggregation, sentence embeddings...
    
class WordEmbedding(Embedding):
    def __init__(self, agg_fn):
        self.agg_fn = agg_fn
    
    def encode(self, tweets):
        if self.embeddings is not None:
            encodings = []
            for tweet in tweets.full_text:
                encoding = np.zeros((self.dim,))
                for sent in tweet:
                    for word in sent:
                        w = self.access_fn(word)
                        if w in self.embeddings:    # Aggregate word embeddings over encoding vector
                            encoding = map(self.agg_fn(encoding, self.embeddings[w])) 
                encodings.append(encoding)
        else:
            raise AttributeError('Embeddings not specified. Use Embedding.extract(pre_trained_emb) first')
        
def access_features(word):
    lemma = word.get_lemma()
    if lemma.startswith('@'):
        return '_@mention'
    elif lemma.startswith('#'):
        return '_#hashtag'
    elif lemma.startswith('https://'):
        return '_url'
    return lemma

if __name__=='__main__':
    tweets = db.import_mongodb('tweets', limit=100)
    tweets = text.preprocess(tweets)
    #fe = BinaryBOW(lambda x: x.get_lemma())
    fe = CountsBOW(lambda x: x.get_lemma())
    result = fe.extract(tweets).encode(tweets)
    print(result)