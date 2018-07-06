from common import feature_extraction
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
import sys


def balanced_acc(true, pred):
    tp,p,tn,n = 0,0,0,0
    for a,b in zip(true,pred):
        if a==1:
            p += 1
            if b==1:
                tp += 1
        else:
            n += 1
            if b!=1:
                tn += 1
    return 0.5 * (tp/p + tn/n)

class WordRelevancePredictor():
    
    def __init__(self, phrases, labels, extractor, folds):
        self.phrases = phrases
        self.X = np.array(extractor.extract(phrases).encode(phrases))
        self.labels = np.array(labels)
        self.X, self.X_t, self.labels, self.labels_t = train_test_split(self.X, self.labels, test_size=0.1)
        self.features_idx = extractor.features_idx
        self.supports = extractor.supports
        self.folds = folds
    
    def compute(self):
        print('Not implemented')
        
    def show(self):
        print('Not implemented')
    
class RelevanceByMutualInfo(WordRelevancePredictor):
    def compute(self):
        smooth = True
        fmax = 100
        alpha = 0.75
        male_results = mutual_info_classif(self.X, self.labels)
        female_results = mutual_info_classif(self.X, [-x for x in self.labels])
        results = {}
        ratios = {}
        for i in range(self.X.shape[1]):
            results[(i,1)] = male_results[i]
            results[(i,-1)] = female_results[i]
            if smooth and self.supports[self.features_idx[i]]<fmax:
                results[(i,1)] = results[(i,1)] * pow(self.supports[self.features_idx[i]]/fmax, alpha)
                results[(i,-1)] = results[(i,-1)] * pow(self.supports[self.features_idx[i]]/fmax, alpha)
            ratios[i] = results[(i,1)]/results[(i,-1)]
        self.results = ratios
        return ratios
    
    def retrieve(self, top=20):
        males = sorted([(self.features_idx[x],v,self.supports[self.features_idx[x]]) for x,v in self.results.items()], key=lambda x:x[1], reverse=True)[:top]
        females = sorted([(self.features_idx[x],v,self.supports[self.features_idx[x]]) for x,v in self.results.items()], key=lambda x:x[1])[:top]
        return males, females
    
    def show(self, top=20, to=sys.stdout):
        males, females = self.retrieve(top)
        print('Male predictors:', file=to)
        print('word : mutual_info', ':', 'support', file=to)
        for (word, value, sup) in males:
            print(word, ':', value, ':', sup, file=to)
        print('\nFemale predictors:', file=to)
        print('word : mutual_info', ':', 'support', file=to)
        for (word, value, sup) in females:
            print(word, ':', value, ':', sup, file=to)

class RelevanceByRegression(WordRelevancePredictor):
    def __init__(self, phrases, labels, extractor=feature_extraction.BinaryBOW(1, lambda x: x.get_lemma()), folds=10, c=1, predname='Ridge (L2) regression'):
        self.predname = predname
        self.c = c
        super().__init__(phrases, labels, extractor, folds)

    def _compute(self, model):
        kf = KFold(n_splits=self.folds)
        self.results = np.array((self.X.shape[1],))
        tr_f1, tr_prec, tr_recall, tr_auc, tr_ba, val_f1, val_prec, val_recall, val_auc, val_ba = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for train_idx, val_idx in kf.split(self.X):
            model = model.fit(self.X[train_idx], self.labels[train_idx])
            tr_preds = model.predict(self.X[train_idx])
            val_preds = model.predict(self.X[val_idx])
            tr_f1 += metrics.f1_score(self.labels[train_idx], tr_preds)
            tr_prec += metrics.precision_score(self.labels[train_idx], tr_preds)
            tr_recall += metrics.recall_score(self.labels[train_idx], tr_preds)
            tr_auc += metrics.roc_auc_score(self.labels[train_idx], tr_preds)
            tr_ba += balanced_acc(self.labels[train_idx], tr_preds)
            val_f1 += metrics.f1_score(self.labels[val_idx], val_preds)
            val_prec += metrics.precision_score(self.labels[val_idx], val_preds)
            val_recall += metrics.recall_score(self.labels[val_idx], val_preds)
            val_auc += metrics.roc_auc_score(self.labels[val_idx], val_preds)
            val_ba += balanced_acc(self.labels[val_idx], val_preds)
            self.results = np.add(self.results, model.coef_.flatten(), casting='unsafe')
        self.results /= self.folds
        tr_f1 /= self.folds
        tr_prec /= self.folds
        tr_recall /= self.folds
        tr_auc /= self.folds
        tr_ba /= self.folds
        val_f1 /= self.folds
        val_prec /= self.folds
        val_recall /= self.folds
        val_auc /= self.folds
        val_ba /= self.folds

        # Let test scores calculated, it is responsibility of the developer not to look at them until a model is selected by validation
        te_preds = model.predict(self.X_t)
        te_f1 = metrics.f1_score(self.labels_t, te_preds)
        te_prec = metrics.precision_score(self.labels_t, te_preds)
        te_recall = metrics.recall_score(self.labels_t, te_preds)
        te_auc = metrics.roc_auc_score(self.labels_t, te_preds)
        te_ba = balanced_acc(self.labels_t, te_preds)

        self.scores = {'Train_F1' : tr_f1, 'Train_Precision': tr_prec, 'Train_Recall': tr_recall, 'Train_AUC': tr_auc, 'Train_Acc': tr_ba, \
                'Validation_F1' : val_f1, 'Validation_Precision': val_prec, 'Validation_Recall': val_recall, 'Validation_AUC': val_auc, 'Validation_Acc': val_ba, \
                'Test_F1': te_f1, 'Test_Precision': te_prec, 'Test_Recall': te_recall, 'Test_AUC': te_auc, 'Test_Acc': te_ba}
        self.females = sum([1 for l in self.labels if l==-1])
        self.males = sum([1 for l in self.labels if l==1])
        self.results = model.coef_.flatten()
        return self.results


    def compute(self):
        return self._compute(LogisticRegression(class_weight='balanced', C=self.c))
    
    def retrieve(self, top=20):
        l = [(self.features_idx[x], v, self.supports[self.features_idx[x]]) for (x,v) in enumerate(self.results)]  # Generator?
        males = sorted(l, key=lambda x:x[1], reverse=True)[:top]
        females = sorted(l, key=lambda x:x[1])[:top]
        return males, females
    
    def show(self, top=20, to=sys.stdout):
        print(self.predname)
        print('Variables used:', str(len(self.features_idx)))
        print('Samples (total):', str(self.males + self.females))
        print('\tClass 1:', str(self.males), str(self.males/(self.males+self.females)))
        print('\tClass 0:', str(self.females), str(self.females/(self.males+self.females)))
        males, females = self.retrieve(top)
        print('Male predictors:', file=to)
        print('word : coef : odds_ratio', file=to)
        for (word, value, sup) in males:
            print(word, ':', value, ':', np.exp(value), ':', sup, file=to)
        print('\nFemale predictors:', file=to)
        print('word : -coef : odds_ratio', file=to)
        for (word, value, sup) in females:
            print(word, ':', -value, ':', np.exp(-value), ':', sup, file=to)

class RelevanceByLassoRegression(RelevanceByRegression):
    def __init__(self, phrases, labels, extractor=feature_extraction.BinaryBOW(1, lambda x: x.get_lemma()), folds=10, c=1):
        super().__init__(phrases, labels, extractor, folds, c, 'Lasso (L1) regression')

    def compute(self):
        return self._compute(LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=self.c))

class RelevanceBySupportVectors(RelevanceByRegression):
    def __init__(self, phrases, labels, extractor=feature_extraction.BinaryBOW(1, lambda x: x.get_lemma()), folds=10, c=1):
        super().__init__(phrases, labels, extractor, folds, c, 'Linear SVM')

    def compute(self):
        return self._compute(LinearSVC(class_weight='balanced', C=self.c))


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

