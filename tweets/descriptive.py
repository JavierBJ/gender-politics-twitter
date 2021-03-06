import numpy as np
from common import mongo
from collections import Counter

def word_frequency_distribution(tweets):
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
    
    return form_distribution, lemma_distribution

class DBDescriptor():
    def __init__(self):
        self.db = mongo.DB()
        self.db_politicians = self.db.import_users_mongodb({'polit':1})
        self.db_tweets_from_polit = self.db.import_tweets_mongodb({'msg':'tweet'})
        self.db_replies = self.db.import_tweets_mongodb({'msg':'reply'})

    def tweets(self):
        return self.db.db['tweets'].find().count()

    def originals(self):
        return self.db.db['tweets'].find({'msg':'tweet'}).count()

    def male_originals(self):
        return self.db.db['tweets'].find({'msg':'tweet', 'author_gender':1}).count()

    def female_originals(self):
        return self.db.db['tweets'].find({'msg':'tweet', 'author_gender':-1}).count()

    def mentions(self):
        return self.db.db['tweets'].find({'msg':'mention'}).count()

    def replies(self):
        return self.db.db['tweets'].find({'msg':'reply'}).count()

    def users(self):
        return self.db.db['users'].find().count()

    def politicians(self):
        return self.db.db['users'].find({'polit':1}).count()

    def individuals(self):
        return self.db.db['users'].find({'polit':0}).count()

    def politicians_per_aut(self):
        autonomies = self.db.db['users'].find({'polit':1}).distinct('autname')
        return {a:self.db.db['users'].find({'autname':a}).count() for a in autonomies}

    def politicians_per_gender(self):
        m = self.db.db['users'].find({'polit':1, 'gender':1}).count()
        f = self.db.db['users'].find({'polit':1, 'gender':-1}).count()
        return m,f

    def individuals_per_gender(self):
        m = self.db.db['users'].find({'polit':0, 'gender':1}).count()
        f = self.db.db['users'].find({'polit':0, 'gender':-1}).count()
        return m,f

    def replies_per_genders(self):
        author_gender = [1, -1, 0]
        receiver_gender = [1, -1, 0]
        results = np.zeros((len(author_gender),len(receiver_gender)))
        for (i,ag) in enumerate(author_gender):
            for (j,rg) in enumerate(receiver_gender):
                results[i,j] = self.db.db['tweets'].find({'msg':'reply', 'author_gender':ag, 'receiver_gender':rg}).count()
        return results

    def _variable_per_gender(self, data, var_desired, var_groupby):
        mean_male = np.mean(data.loc[data[var_groupby]==1, var_desired])
        mean_female = np.mean(data.loc[data[var_groupby]==-1, var_desired])
        return mean_male, mean_female

    def _user_variable_per_gender(self, varname):
        return self._variable_per_gender(self.db_politicians, varname, 'gender')

    def _tweet_variable_per_gender(self, varname):
        return self._variable_per_gender(self.db_tweets_from_polit, varname, 'author_gender')

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
            results[(f_gen, t_gen)] = len(self.db_replies[(self.db_replies['author_gender']==f_gen) & (self.db_replies['receiver_gender']==t_gen)])
        return results

