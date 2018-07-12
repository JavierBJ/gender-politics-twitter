import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix

class FeatureExtractor():
    def __init__(self, extractors):
        self.extractors = extractors
        
    def extract(self, source):
        for ext in self.extractors:
            ext.extract(source)
            
        self.features = {}
        self.features_idx = []
        for ext in self.extractors:
            self.features.update(ext.features)
            self.features_idx.extend(ext.features_idx)
        return self
        
    def encode(self, tweets):
        codes = []
        for ext in self.extractors:
            codes.append(ext.encode(tweets))
        newcodes = []
        for i in range(len(codes[0])):
            newcode = [codes[j][i] for j in range(len(codes))]
            newcode = np.concatenate(newcode).ravel().tolist()
            newcodes.append(newcode)
            
        return np.array(newcodes)

class FeatureExtractorBOW(FeatureExtractor):
    def __init__(self, n, inc, top, access_fn, access_word=lambda x:x, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        if top<1:
            top = float('inf')  # No top limit if not positive
        self.n = n
        self.inc = inc
        self.top = top
        self.keep_words_freq = keep_words_freq
        self.keep_words_rank = keep_words_rank
        self.remove_stopwords = remove_stopwords
        self.access_fn = access_fn
        self.features = None
        self.access_word = access_word
        self.field = field
    
    def extract(self, source):
        # Filter out unnecessary tokens from tokenized tweets
        tweets_filtered = []
        for tweet in source[self.field]:
            tweets_filtered.append([token for sent in tweet for token in sent if not (self._is_sw(self.access_word(self.access_fn(token))) 
                                                    or self._is_punct(self.access_word(self.access_fn(token))) 
                                                    or self._is_short(self.access_word(self.access_fn(token))))])
        self.tweets_filtered = tweets_filtered
        
        # Elaborate dictionary of n-grams
        self.grams = self._elaborate_ngrams()
        
        # Cut out dictionary of n-grams
        self.grams = self._cut_out()
            
        # Create variables for use of features
        self.features = {x:i for i,(x,_) in enumerate(self.grams)}
        self.features_idx = [w for w,_ in self.grams]
        self.supports = {x:s for x,s in self.grams}
        
        return self
    
    def _elaborate_ngrams(self):
        grams = Counter()
        for tweet in self.tweets_filtered:
            grams.update({tuple([self.access_fn(tweet[j]) for j in range(i,i+self.n)]) for i in range(len(tweet)-self.n+1)})
        return grams
    
    def _cut_out(self):
        grams = self.grams
        if self.keep_words_rank is not None and self.keep_words_rank>0:
            grams = grams.most_common(self.keep_words_rank)
        elif self.keep_words_freq is not None and self.keep_words_freq>0:
            grams = [(w,v) for w,v in grams.items() if v>=self.keep_words_freq]
        else:
            grams = grams.most_common()
        return grams
    
    def _is_sw(self, w):
        return self.remove_stopwords and w.lower() in stopwords.words('spanish')

    def _is_punct(self, w):
        from string import punctuation
        for p in punctuation:
            w = w.replace(p,'')
        for n in '0123456789':
            w = w.replace(n,'')
        return self._is_short(w)

    def _is_short(self, w):
        import emoji
        if w.startswith('@') or w.startswith('#') or w.startswith('https://'):
            return True
        elif len(w)==1:
            return w[0] not in emoji.UNICODE_EMOJI
        else:
            return len(w)<3

    def _vectorize(self, tweets):
        row = []
        col = []
        data = []
        for (r, tweet) in enumerate(tweets[self.field]):
            for sent in tweet:
                for i in range(len(sent)-self.n+1):
                    g = tuple([self.access_fn(sent[i]) for i in range(i, i+self.n)])
                    if g in self.features:
                        c = self.features[g]
                        row.append(r)
                        col.append(c)
                        data.append(self.inc)
        sp = coo_matrix((data, (row,col)), shape=(len(tweets[self.field]), len(self.features)))
        sp.sum_duplicates()
        # Cut off to the maximum number of counts declared
        for i in range(sp.data.shape[0]):
            sp.data[i] = min(sp.data[i], self.top)
        return sp.tocsr()
        if self.features is not None:
            encodings = []
            for tweet in tweets[self.field]:
                encoding = np.zeros((len(self.features),))  # Prepare encoding vector
                for sent in tweet:
                    for i in range(len(sent)-self.n+1):
                        g = tuple([self.access_fn(sent[i]) for i in range(i,i+self.n)])
                        # Only increment counter if not already in the top
                        if g in self.features and encoding[self.features[g]]<self.top:
                            encoding[self.features[g]] += self.inc
                encodings.append(encoding)
        else:
            raise AttributeError('Features not extracted. Use FeatureExtractor.extract(source) first')
        return encodings

    def encode(self, tweets):
        return self._vectorize(tweets)

class BinaryBOW(FeatureExtractorBOW):
    def __init__(self, n, access_fn, access_word = lambda x:x, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, 1,1, access_fn, access_word, keep_words_freq, keep_words_rank, remove_stopwords, field)

class CountsBOW(FeatureExtractorBOW):
    def __init__(self, n, access_fn, access_word = lambda x:x, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, 1,-1, access_fn, access_word, keep_words_freq, keep_words_rank, remove_stopwords, field)

class TfIdfBOW(CountsBOW):
    def __init__(self, n, access_fn, access_word = lambda x:x, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, access_fn, access_word, keep_words_freq, keep_words_rank, remove_stopwords, field)

    def encode(self, tweets):
        counts = self._vectorize(tweets)
        return TfidfTransformer().fit_transform(counts).todense()

class FeatureExtractorBOWGender(FeatureExtractorBOW):
    def __init__(self, n, inc, top, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, inc, top, lambda x: self._tag_by_gender(x.get_lemma(), x.get_tag()), lambda x: x[0], keep_words_freq, keep_words_rank, remove_stopwords, field)

    def _tag_by_gender(self, lemma, tag):
        if tag[0] in 'ADP':
            return (lemma, tag[3])
        elif tag[0]=='N':
            return (lemma, tag[2])
        else:
            return (lemma, '0')

class BinaryBOWGender(FeatureExtractorBOWGender):
    def __init__(self, n, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, 1,1, keep_words_freq, keep_words_rank, remove_stopwords, field)

class CountsBOWGender(FeatureExtractorBOWGender):
    def __init__(self, n, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, 1,-1, keep_words_freq, keep_words_rank, remove_stopwords, field)

class TfIdfBOWGender(CountsBOWGender):
    def __init__(self, n, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, keep_words_freq, keep_words_rank, remove_stopwords, field)

    def encode(self, tweets):
        counts = self._vectorize(tweets)
        return TfidfTransformer().fit_transform(counts).todense()

class FeatureExtractorPOS(FeatureExtractorBOW):
    def __init__(self, inc, top, field='full_text'):
        super().__init__(inc, top, lambda x: self._analyze_tag(x.get_tag()), field)
        
    def extract(self, source):
        tags = set()
        for tweet in source[self.field]:
            for sent in tweet:
                for word in sent:
                    tags.update(self.access_fn(word))
        
        self.features = {x:i for i,x in enumerate(tags)}
        self.features_idx = list(tags)
        
        return self
    
    def encode(self, tweets):
        if self.features is not None:
            encodings = []
            for tweet in tweets[self.field]:
                encoding = np.zeros((len(self.features),))  # Prepare encoding vector
                for sent in tweet:
                    for word in sent:
                        ws = self.access_fn(word)
                        for w in ws:
                            # Only increment counter if not already in the top
                            if w in self.features and encoding[self.features[w]]<self.top:
                                encoding[self.features[w]] += self.inc
                encodings.append(encoding)
        else:
            raise AttributeError('Features not extracted. Use FeatureExtractor.extract(source) first')
        return encodings
    
    def _analyze_tag(self, code):
        tags = set()
        if code[0]=='A':
            tags.add('Adjective')
        elif code[0]=='D':
            tags.add('Determiner')
        elif code[0]=='N':
            tags.add('Noun')
            if code[1]=='C':
                tags.add('Noun, Common')
            elif code[1]=='P':
                tags.add('Noun, Proper')
        elif code[0]=='P':
            tags.add('Pronoun')
        elif code[0]=='R':
            tags.add('Adverb')
        elif code[0]=='V':
            tags.add('Verb')
            if code[2]=='I':
                tags.add('Verb, Indicative')
            elif code[2]=='S':
                tags.add('Verb, Subjunctive')
            elif code[2]=='M':
                tags.add('Verb, Imperative')
            elif code[2]=='P':
                tags.add('Verb, Participle')
            elif code[2]=='G':
                tags.add('Verb, Gerund')
            elif code[2]=='N':
                tags.add('Verb, Infinitive')
                
            if code[3]=='P':
                tags.add('Verb, Present')
            elif code[3]=='I':
                tags.add('Verb, Imperfect')
            elif code[3]=='F':
                tags.add('Verb, Future')
            elif code[3]=='S':
                tags.add('Verb, Past')
            elif code[3]=='C':
                tags.add('Verb, Conditional')
        elif code[0]=='Z':
            tags.add('Number')
        elif code[0]=='W':
            tags.add('Date')
        elif code[0]=='I':
            tags.add('Interjection')
        
        index_gender = -1
        if code[0]=='A' or code[0]=='D' or code[0]=='P':
            index_gender = 3
        elif code[0]=='N':
            index_gender = 2
        elif code[0]=='V':
            index_gender = 6
        if index_gender!=-1:
            if code[index_gender]=='F':
                tags.add('Feminine Gender')
            elif code[index_gender]=='M':
                tags.add('Masculine Gender')
            elif code[index_gender]=='C':
                tags.add('Common Gender')
        
        index_number = -1
        if code[0]=='A' or code[0]=='D' or code[0]=='P':
            index_number = 4
        elif code[0]=='N':
            index_number = 3
        elif code[0]=='V':
            index_number = 5
        if index_number!=-1:
            if code[index_number]=='S':
                tags.add('Singular Number')
            elif code[index_number]=='P':
                tags.add('Plural Number')
            elif code[index_number]=='N':
                tags.add('Invariable Number')
                    
        index_person = -1
        if code[0]=='D' or code[0]=='P':
            index_person = 2
        elif code[0]=='V':
            index_person = 4
        if index_person!=-1:
            if code[index_person]=='1':
                tags.add('1st Person')
            elif code[index_person]=='2':
                tags.add('2nd Person')
            elif code[index_person]=='3':
                tags.add('3rd Person')
        
        return tags
    
class BinaryPOS(FeatureExtractorPOS):
    def __init__(self):
        super().__init__(1,1)

class CountsPOS(FeatureExtractorPOS):
    def __init__(self):
        super().__init__(1,-1)

class FeatureExtractorChars(FeatureExtractorBOW):
    def __init__(self, n, inc, top, access_fn=lambda x:x.get_form(), access_word=lambda x:x, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, inc, top, access_fn, access_word, keep_words_freq, keep_words_rank, remove_stopwords, field)
    
    def _elaborate_ngrams(self):
        grams = Counter()
        for tweet in self.tweets_filtered:
            tweet_text = ' '.join([self.access_fn(token) for token in tweet])   # Sentence separation ommited at text-line creation
            grams.update({tuple([tweet_text[j] for j in range(i,i+self.n)]) for i in range(len(tweet_text)-self.n+1)})
        return grams
    
    def _vectorize(self, tweets):
        row = []
        col = []
        data = []
        for (r, tweet) in enumerate(tweets[self.field]):
            tweet_text = ' '.join([self.access_fn(token) for sent in tweet for token in sent])
            for i in range(len(tweet_text)-self.n+1):
                g = tuple([tweet_text[i] for i in range(i, i+self.n)])
                if g in self.features:
                    c = self.features[g]
                    row.append(r)
                    col.append(c)
                    data.append(1)
        sp = coo_matrix((data, (row, col)), shape=(len(tweets[self.field]), len(self.features)))
        return sp.tocsr()
        if self.features is not None:
            encodings = []
            for tweet in tweets[self.field]:
                tweet_text = ' '.join([self.access_fn(token) for sent in tweet for token in sent])
                encoding = np.zeros((len(self.features),))  # Prepare encoding vector
                for i in range(len(tweet_text)-self.n+1):
                    g = tuple([tweet_text[i] for i in range(i,i+self.n)])
                    # Only increment counter if not already in the top
                    if g in self.features and encoding[self.features[g]]<self.top:
                        encoding[self.features[g]] += self.inc
                encodings.append(encoding)
        else:
            raise AttributeError('Features not extracted. Use FeatureExtractor.extract(source) first')
        return encodings

    def encode(self, tweets):
        return np.array(self._vectorize(tweets))

class BinaryChars(FeatureExtractorChars):
    def __init__(self, n, access_fn=lambda x:x.get_form(), access_word=lambda x:x, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, 1, 1, access_fn, access_word, keep_words_freq, keep_words_rank, remove_stopwords, field)
        
class CountsChars(FeatureExtractorChars):
    def __init__(self, n, access_fn=lambda x:x.get_form(), access_word=lambda x:x, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, 1, -1, access_fn, access_word, keep_words_freq, keep_words_rank, remove_stopwords, field)

class TfIdfChars(CountsChars):
    def __init__(self, n, access_fn=lambda x:x.get_form(), access_word=lambda x:x, keep_words_freq=0, keep_words_rank=0, remove_stopwords=False, field='full_text'):
        super().__init__(n, access_fn, access_word, keep_words_freq, keep_words_rank, remove_stopwords, field)

    def encode(self, tweets):
        counts = self._vectorize(tweets)
        return TfidfTransformer().fit_transform(counts).todense()

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
                    for w in sent:
                        if w in self.embeddings:    # Aggregate word embeddings over encoding vector
                            encoding = map(self.agg_fn(encoding, self.embeddings[w])) 
                encodings.append(encoding)
        else:
            raise AttributeError('Embeddings not specified. Use Embedding.extract(pre_trained_emb) first')

