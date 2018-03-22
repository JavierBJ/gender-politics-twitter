import pandas as pd
from common import text
from data import mongo
from collections import Counter
import matplotlib.pyplot as plt

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


if __name__=='__main__':
    tweets = db.import_mongodb('tweets', limit=1000)
    tweets = text.preprocess(tweets)
    
    word_frequency_distribution(tweets.full_text, plot=True)
