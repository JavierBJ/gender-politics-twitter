from common import text, feature_extraction, mongo
import sys, os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

score_names = {'tr_f1':'Train_F1', 'val_f1':'Validation_F1', 'te_f1':'Test_F1', \
               'tr_prec':'Train_Precision', 'val_prec':'Validation_Precision', 'te_prec':'Test_Precision', \
               'tr_rec':'Train_Recall', 'val_rec':'Validation_Recall', 'te_rec':'Test_Recall', \
               'tr_auc':'Train_AUC', 'val_auc':'Validation_AUC', 'te_auc':'Test_AUC'}

class LanguageClassifier():
    def __init__(self, model, extractor, X, target, score, folds):
        y = np.array(target).reshape((-1,))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self._feature_extraction(X_train, X_test, y_train, y_test, extractor)
        self.model = model
        self.score = score
        self.folds = folds
    
    def _feature_extraction(self, X_train, X_test, y_train, y_test, extractor):
        print('Extracting features...')
        self.extractor = extractor.extract(X_train)
        print('Extracted train features')
        self.X = extractor.encode(X_train)
        print('Encoded train data as features')
        self.y = y_train
        self.X_test = extractor.encode(X_test)
        print('Encoded train data as features')
        self.y_test = y_test
        print('\tFeatures extracted.')
    
    def compute(self):
        kf = KFold(n_splits=self.folds)
        self.results = np.array((self.X.get_shape()[1],))
        tr_f1, tr_prec, tr_recall, tr_auc, val_f1, val_prec, val_recall, val_auc = 0, 0, 0, 0, 0, 0, 0, 0
        for train_idx, val_idx in kf.split(self.X):
            model = self.model.fit(self.X[train_idx], self.y[train_idx])
            tr_preds = self.model.predict(self.X[train_idx])
            val_preds = self.model.predict(self.X[val_idx])
            tr_f1 += metrics.f1_score(self.y[train_idx], tr_preds)
            tr_prec += metrics.precision_score(self.y[train_idx], tr_preds)
            tr_recall += metrics.recall_score(self.y[train_idx], tr_preds)
            tr_auc += metrics.roc_auc_score(self.y[train_idx], tr_preds)
            val_f1 += metrics.f1_score(self.y[val_idx], val_preds)
            val_prec += metrics.precision_score(self.y[val_idx], val_preds)
            val_recall += metrics.recall_score(self.y[val_idx], val_preds)
            val_auc += metrics.roc_auc_score(self.y[val_idx], val_preds)
            self.results = np.add(self.results, model.coef_.flatten(), casting='unsafe')
        self.results /= self.folds
        tr_f1 /= self.folds
        tr_prec /= self.folds
        tr_recall /= self.folds
        tr_auc /= self.folds
        val_f1 /= self.folds
        val_prec /= self.folds
        val_recall /= self.folds
        val_auc /= self.folds

        # Re-train with the whole training set
        model = self.model.fit(self.X, self.y)
        self.results = model.coef_.flatten()
        
        # Let test scores calculated, though they must not be observed during fine-tuning
        te_preds = self.model.predict(self.X_test)
        self.te_preds = te_preds
        te_f1 = metrics.f1_score(self.y_test, te_preds)
        te_prec = metrics.precision_score(self.y_test, te_preds)
        te_recall = metrics.recall_score(self.y_test, te_preds)
        te_auc = metrics.roc_auc_score(self.y_test, te_preds)
        
        self.conf = metrics.confusion_matrix(self.y_test, te_preds)
        self.scores = {'Train_F1' : tr_f1, 'Train_Precision': tr_prec, 'Train_Recall': tr_recall, 'Train_AUC': tr_auc, \
                'Validation_F1' : val_f1, 'Validation_Precision': val_prec, 'Validation_Recall': val_recall, 'Validation_AUC': val_auc, \
                'Test_F1': te_f1, 'Test_Precision': te_prec, 'Test_Recall': te_recall, 'Test_AUC': te_auc}
        return self.scores
        
    def describe(self):
        print('         | Train | Test')
        print('Class 1  |', sum([1 for t in self.y if t==1]), sum([1 for t in self.y_test if t==1]))
        print('Class 0  |', sum([1 for t in self.y if t==-1]), sum([1 for t in self.y_test if t==-1]))

class LanguageAnalysisClassifier(LanguageClassifier):
    def retrieve(self, top=20):
        l = [(self.extractor.features_idx[x], v, self.extractor.supports[self.extractor.features_idx[x]]) for (x,v) in enumerate(self.results)]
        males = sorted(l, key=lambda x:x[1], reverse=True)[:top]
        females = sorted(l, key=lambda x:x[1])[:top]
        return males, females
    
    def show(self, top=20, to=sys.stdout):
        males, females = self.retrieve(top)
        print('Male predictors:', file=to)
        print('word : coef : odds_ratio', file=to)
        for (word, value, sup) in males:
            print(word, ':', value, ':', np.exp(value), ':', sup, file=to)
        print('\nFemale predictors:', file=to)
        print('word : -coef : odds_ratio', file=to)
        for (word, value, sup) in females:
            print(word, ':', -value, ':', np.exp(-value), ':', sup, file=to)

def analyze(who='author', alg='lasso', score='val_auc', dbname='gender', limit=0, kfolds=10, sw=True, kwf=None, kwr=5000, alpha=1):
    # Craft path for outputs from parameters
    path = 'results/analysis/'
    expname = who+'_'+alg
    if kwf is not None:
        expname += '_freq' + str(kwf)
    if kwr is not None:
        expname += '_rank' + str(kwr)
    if sw:
        expname += '_sw'
    path += expname
    if not os.path.exists(path):
        os.makedirs(path)
    log = open(path + '/log.txt', 'w')
    
    # Translates short names of scores into long names understood by the model
    if score not in score_names.values():
        score = score_names[score]
    
    db = mongo.DB(dbname)
    if who=='author':
        tweets_df = db.import_tagged_by_author_gender_political_tweets_mongodb(weeks=None, limit=limit)
        tweets, labels = text.preprocess(tweets_df, 'author_gender')
    elif who=='receiver':
        tweets_df = db.import_tagged_by_receiver_gender_tweets_mongodb(limit=limit)
        tweets, labels = text.preprocess(tweets_df, 'receiver_gender')
    
    ext = feature_extraction.BinaryBOW(1, lambda x:x[1], keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
    
    best = 0
    for a in alpha:
        c_reg = 1/a
        if alg in ['lasso', 'l1']:
            model = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=c_reg)
        elif alg in ['ridge', 'l2']:
            model = LogisticRegression(class_weight='balanced', C=c_reg)
        elif alg=='svm':
            model = LinearSVC(class_weight='balanced', C=c_reg)
        analyzer = LanguageAnalysisClassifier(model, ext, tweets, labels, score, kfolds)
        res = analyzer.compute()
        os.makedirs(path, exist_ok=True)
        print('Experiment', expname, ', case', a, '...', file=log)
        analyzer.show(top=100, to=open(path+'/'+'a'+str(a).replace('.','_')+'.txt', 'w'))
        print('Train - Validation', file=log)
        print('F1:', res['Train_F1'], '-', res['Validation_F1'], file=log)
        print('Precision:', res['Train_Precision'], '-', res['Validation_Precision'], file=log)
        print('Recall:', res['Train_Recall'], '-', res['Validation_Recall'], file=log)
        print('AUC:', res['Train_AUC'], '-', res['Validation_AUC'], file=log)
        print('', file=log)
        
        if res[analyzer.score]>best:
            best = res[analyzer.score]
            best_analyzer = analyzer
            best_a = a
            
    print('Best experiment: Alpha =', best_a, file=log)
    print('Test scores', file=log)
    print('F1:', best_analyzer.scores['Test_F1'], file=log)
    print('Precision:', best_analyzer.scores['Test_Precision'], file=log)
    print('Recall:', best_analyzer.scores['Test_Recall'], file=log)
    print('AUC:', best_analyzer.scores['Test_AUC'], file=log)

def detect(dv='hostility', alg='lasso', prep='lemma', how='tfidf', dbname='gender', limit=0, kfolds=10, sw=True, kwf=None, kwr=5000, w=1, c=0, alpha=1, hidden=500, learningrate=0.0001, maxfeatures=1.0, minsamplesleaf=1):
    db = mongo.DB(dbname)
    tweets_df = db.import_tagged_by_hostile_tweets_mongodb(limit=limit)
    tweets_df = text.preprocess(tweets_df)
    
    if dv=='hostility':
        label = 'is_hostile'
    elif dv=='sexism':
        label = 'is_sexist'
    elif dv=='gender':
        label = 'author_gender'
    
    if prep=='lemma':
        f_prep = lambda x:x[1]
    elif prep=='form':
        f_prep = lambda x:x[0]
    
    if w:
        if how=='binary':
            ext = feature_extraction.BinaryBOW(w, f_prep, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
        elif how=='counts':
            ext = feature_extraction.CountsBOW(w, f_prep, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
        elif how=='tfidf':
            ext = feature_extraction.TfIdfBOW(w, f_prep, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
    elif c:
        if how=='binary':
            ext = feature_extraction.BinaryChars(c, f_prep, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
        elif how=='counts':
            ext = feature_extraction.CountsChars(c, f_prep, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
        elif how=='tfidf':
            ext = feature_extraction.TfIdfChars(c, f_prep, keep_words_freq=kwf, keep_words_rank=kwr, remove_stopwords=sw)
    
    if alg in ['lasso', 'l1']:
        for a in alpha:
            c_reg = 1/a
            model = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=c_reg)
            det = LanguageClassifier(model, ext, tweets_df, label, None, kfolds)
            res = det.compute()
    elif alg in ['ridge', 'l2']:
        for a in alpha:
            c_reg = 1/a
            model = LogisticRegression(class_weight='balanced', C=c_reg)
            det = LanguageClassifier(model, ext, tweets_df, label, None, kfolds)
            res = det.compute()
    elif alg=='svm':
        for a in alpha:
            c_reg = 1/a
            model = SVC(class_weight='balanced', C=c_reg)
            det = LanguageClassifier(model, ext, tweets_df, label, None, kfolds)
            res = det.compute()
    elif alg=='nb':
        model = BernoulliNB()
        det = LanguageClassifier(model, ext, tweets_df, label, None, kfolds)
        res = det.compute()
    elif alg=='mlp':
        for lr in learningrate:
            for h in hidden:
                for a in alpha:
                    model = MLPClassifier(hidden_layer_sizes=(h,), learning_rate_init=lr, alpha=a)
                    det = LanguageClassifier(model, ext, tweets_df, label, None, kfolds)
                    res = det.compute()
    elif alg=='rf':
        for mf in maxfeatures:
            for lf in minsamplesleaf:
                model = RandomForestClassifier(n_estimators=50, max_features=mf, min_samples_leaf=lf)
                det = LanguageClassifier(model, ext, tweets_df, label, None, kfolds)
                res = det.compute()
    return 0

