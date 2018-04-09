import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from common import text

class Detector():
    def __init__(self, model, extractor, dataset, target, params, score):
        y = dataset[target]
        X = text.preprocess(dataset)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.extractor = extractor.extract(X_train)
        self.X_train = np.array(extractor.encode(X_train))
        self.y_train = y_train
        self.X_test = np.array(extractor.encode(X_test))
        self.y_test = y_test
        print(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape)
        
        cv = GridSearchCV(model, params, scoring=score, cv=5)
        self.model = model
        self.cv = cv.fit(self.X_train, self.y_train)
    
    def describe(self):
        print('        | Train | Test')
        print('Males   |', sum([1 for t in self.y_train if t==1]), sum([1 for t in self.y_test if t==1]))
        print('Females |', sum([1 for t in self.y_train if t==-1]), sum([1 for t in self.y_test if t==-1]))

    def metrics(self, X_test=None, y_test=None):
        # Uses test specified at creation unless another is passed
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        y_pred = self.predict(X_test)
        
        met = {'accuracy': metrics.accuracy_score(y_test, y_pred),
               'precision': metrics.precision_score(y_test, y_pred),
               'recall': metrics.recall_score(y_test, y_pred), 
               'f1': metrics.f1_score(y_test, y_pred), 
               'roc_auc': metrics.roc_auc_score(y_test, y_pred)}
        
        cv_all = self.cv.cv_results_
        
        cv_res = {'best_estimator': self.cv.best_estimator_,
                  'best_score': self.cv.best_score_,
                  'best_params': self.cv.best_params_}
            
        return met, cv_all, cv_res
    
    def predict(self, X_test=None):
        # Uses test specified at creation unless another is passed
        if X_test is None:
            X_test = self.X_test
        return self.cv.predict(X_test)
