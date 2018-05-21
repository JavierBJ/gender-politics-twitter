import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from common import text
from sexism import detection
from data import mongo
from common import feature_extraction
from sklearn.linear_model import LogisticRegression
import pandas as pd

class Detector():
    def __init__(self, model, extractor, dataset, target, params, score):
        y = dataset[target]
        X = text.preprocess(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self._feature_extraction(X_train, X_test, y_train, y_test, extractor)
        
        return self

    def compute():
        kf = KFold(n_splits=10)
        self.results = np.zeros((self.X.shape[1],))
        tr_f1, tr_prec, tr_recall, tr_auc, val_f1, val_prec, val_recall, val_auc = 0, 0, 0, 0, 0, 0, 0, 0
        for train_idx, val_idx in kf.split(self.X):
            model = model.fit(self.X[train_idx], self.labels[train_idx])
            tr_preds = model.predict(self.X[train_idx])
            val_preds = model.predict(self.X[val_idx])
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

        # Let test scores calculated, it is responsibility of the developer not to look at them until a model is selected by validation
        te_preds = model.predict(self.X_test)
        te_f1 = metrics.f1_score(self.y_test, te_preds)
        te_prec = metrics.precision_score(self.y_test, te_preds)
        te_recall = metrics.recall_score(self.y_test, te_preds)
        te_auc = metrics.roc_auc_score(self.y_test, te_preds)

        self.scores = {'Train_F1' : tr_f1, 'Train_Precision': tr_prec, 'Train_Recall': tr_recall, 'Train_AUC': tr_auc, \
                'Validation_F1' : val_f1, 'Validation_Precision': val_prec, 'Validation_Recall': val_recall, 'Validation_AUC': val_auc, \
                'Test_F1': te_f1, 'Test_Precision': te_prec, 'Test_Recall': te_recall, 'Test_AUC': te_auc}
        return self.results

    def _feature_extraction(self, X_train, X_test, y_train, y_test, extractor):
        self.extractor = extractor.extract(X_train)
        self.X = np.array(extractor.encode(X_train))
        self.y = y_train
        self.X_test = np.array(extractor.encode(X_test))
        self.y_test = y_test
    
    def describe(self):
        print('         | Train | Test')
        print('Class 1  |', sum([1 for t in self.y if t==1]), sum([1 for t in self.y_test if t==1]))
        print('Class 0  |', sum([1 for t in self.y if t==0]), sum([1 for t in self.y_test if t==0]))


class MacroDetector(Detector):
    def __init__(self, detectors, score):
        for s in detectors[0].metrics()[0].keys():
            dets_per_score = {i:d.metrics()[0][s] for i,d in enumerate(detectors)}
            print(s, dets_per_score)
            if s==score:
                dets_scoring = dets_per_score
                print('\t^This is the evaluation metric')
        # Get detector with max score
        print(dets_per_score)
        best = detectors[max(dets_per_score, key=dets_per_score.get)]
        
        # Necessary for the inherited functions to work
        # In essence, MacroDetector looks for the best detector and mutes into it
        self.X_train = best.X_train
        self.X_test = best.X_test
        self.y_train = best.y_train
        self.y_test = best.y_test
        self.model = best.model
        self.cv = best.cv

if __name__=='__main__':
    db = mongo.DB()
    dataset = db.import_tagged_by_author_gender_political_tweets_mongodb(limit=1000)
    target = 'author_gender'
    model = LogisticRegression()
    extractor = feature_extraction.BinaryBOWGender(keep_words_rank=50, remove_stopwords=True)
    params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
    score = 'roc_auc'
    det = detection.Detector(model, extractor, dataset, target, params, score)
    
    mc = MacroDetector([det], 'roc_auc')

