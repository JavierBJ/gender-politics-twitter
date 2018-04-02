import pandas as pd
import numpy as np
from common import text
from data import mongo
from collections import Counter
#import matplotlib.pyplot as plt

def word_frequency_distribution(tweets, plot=False):
    form_counts = Counter()
    lemma_counts = Counter()
    
    for tweet in tweets:
        for sent in tweet:
            form_counts.update([w.get_form() for w in sent])
            lemma_counts.update([w.get_lemma() for w in sent]) 
            
    print('Number of different words:', len(form_counts))
    print('Number of different lemmas:', len(lemma_counts))
    form_distribution = form_counts.most_common()
    lemma_distribution = lemma_counts.most_common()
    
    print('10 Most Common Words:')
    for i in range(10):
        print('\t',lemma_distribution[i][0], '-', lemma_distribution[i][1])
    
    if plot:
        x = range(len(form_distribution))
        plt.plot(x, [y for _,y in form_distribution])
        plt.xticks(x, [t for t,_ in form_distribution], rotation=45)
        plt.show()
        
        x = range(len(lemma_distribution))
        plt.plot(x, [y for _,y in lemma_distribution])
        plt.xticks(x, [t for t,_ in lemma_distribution], rotation=45)
        plt.show()
    
    return form_distribution, lemma_distribution

class DBDescriptor():
    def __init__(self):
        self.db = mongo.DB()
        self.politicians = self.db.import_users_mongodb({'polit':1})
        self.tweets_from_polit = self.db.import_tweets_mongodb({'msg':'tweet'})
        self.replies = self.db.import_tweets_mongodb({'msg':'reply'})

    def _variable_per_gender(self, data, var_desired, var_groupby):
        mean_male = np.mean(data.loc[data[var_groupby]==1, var_desired])
        mean_female = np.mean(data.loc[data[var_groupby]==-1, var_desired])
        return mean_male, mean_female

    def _user_variable_per_gender(self, varname):
        return self._variable_per_gender(self.politicians, varname, 'gender')

    def _tweet_variable_per_gender(self, varname):
        return self._variable_per_gender(self.tweets_from_polit, varname, 'author_gender')

    def followers_per_gender(self):
        return self._user_variable_per_gender('followers_count')

    def favorites_received_per_gender(self):
        return self._tweet_variable_per_gender('favorite_count')

    def retweets_received_per_gender(self):
        return self._tweet_variable_per_gender('retweet_count')

    def contingency_table(self):
        from_gender = [1,1,-1,-1]
        to_gender = [1,-1,1,-1]
        results = {}
        for f_gen, t_gen in zip(from_gender, to_gender):
            results[(f_gen, t_gen)] = len(self.replies[(self.replies['author_gender']==f_gen) & (self.replies['receiver_gender']==t_gen)])
        return results

if __name__=='__main__':
    tweets = db.import_mongodb('tweets', limit=1000)
    tweets = text.preprocess(tweets)
    
    word_frequency_distribution(tweets.full_text, plot=True)
