import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from common import text, feature_extraction, mongo
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class Detector():
    def __init__(self, model, extractor, X, target, params, score, folds):
        y = np.array(X[target]).reshape((-1,))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self._feature_extraction(X_train, X_test, y_train, y_test, extractor)
        self.model = model
        self.folds = folds

    def compute(self):
        kf = KFold(n_splits=self.folds)
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
        tr_f1 /= self.folds
        tr_prec /= self.folds
        tr_recall /= self.folds
        tr_auc /= self.folds
        val_f1 /= self.folds
        val_prec /= self.folds
        val_recall /= self.folds
        val_auc /= self.folds

        # Let test scores calculated, it is responsibility of the developer not to look at them until a model is selected by validation
        model = self.model.fit(self.X, self.y)
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

    def _feature_extraction(self, X_train, X_test, y_train, y_test, extractor):
        self.extractor = extractor.extract(X_train)
        self.X = extractor.encode(X_train)
        self.y = y_train
        self.text_test = X_test['full_text'].tolist()
        self.author_genders = X_test['author_gender'].tolist()
        self.receiver_genders = X_test['receiver_gender'].tolist()
        self.X_test = extractor.encode(X_test)
        self.y_test = y_test
    
    def describe(self):
        print('         | Train | Test')
        print('Class 1  |', sum([1 for t in self.y if t==1]), sum([1 for t in self.y_test if t==1]))
        print('Class 0  |', sum([1 for t in self.y if t==-1]), sum([1 for t in self.y_test if t==-1]))

def detect(dv='hostility', alg='lasso', prep='lemma', how='tfidf', dbname='gender', limit=0, kfolds=10, sw=True, kwf=None, kwr=5000, w=1, c=0, alpha=1, hidden=500, learningrate=0.0001, maxfeatures=1.0, minsamplesleaf=1):
    db = mongo.DB(dbname)
    tweets_df = db.import_tagged_by_hostile_tweets_mongodb(weeks=None, limit=limit)
    tweets_df = text.preprocess(tweets_df)
    
    if dv=='hostility':
        label = 'is_hostile'
    elif dv=='sexism':
        label = 'is_sexist'
    elif dv=='gender':
        label = 'author_gender'
    
    if prep=='lemma':
        f_prep = lambda x: x.get_lemma()
    elif prep=='form':
        f_prep = lambda x: x.get_form()
    
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
            det = Detector(model, ext, tweets_df, label, None, None, kfolds)
            res = det.compute()
    elif alg in ['ridge', 'l2']:
        for a in alpha:
            c_reg = 1/a
            model = LogisticRegression(class_weight='balanced', C=c_reg)
            det = Detector(model, ext, tweets_df, label, None, None, kfolds)
            res = det.compute()
    elif alg=='svm':
        for a in alpha:
            c_reg = 1/a
            model = SVC(class_weight='balanced', C=c_reg)
            det = Detector(model, ext, tweets_df, label, None, None, kfolds)
            res = det.compute()
    elif alg=='nb':
        model = BernoulliNB()
        det = Detector(model, ext, tweets_df, label, None, None, kfolds)
        res = det.compute()
    elif alg=='mlp':
        for lr in learningrate:
            for h in hidden:
                for a in alpha:
                    model = MLPClassifier(hidden_layer_sizes=(h,), learning_rate_init=lr, alpha=a)
                    det = Detector(model, ext, tweets_df, label, None, None, kfolds)
                    res = det.compute()
    elif alg=='rf':
        for mf in maxfeatures:
            for lf in minsamplesleaf:
                model = RandomForestClassifier(n_estimators=50, max_features=mf, min_samples_leaf=lf)
                det = Detector(model, ext, tweets_df, label, None, None, kfolds)
                res = det.compute()
    return 0

