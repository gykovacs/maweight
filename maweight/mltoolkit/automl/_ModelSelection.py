import numpy as np

from ._regressors import *
from maweight.mltoolkit.optimization import EvolutionaryAlgorithm, SimulatedAnnealing
from maweight.mltoolkit.base import VerboseLoggingMixin
from maweight.mltoolkit.automl import FeatureSelectionRegressor, R2_score

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
import sklearn.preprocessing as preprocessing

from sklearn.ensemble import BaggingRegressor
import tqdm

import copy
import os

__all__=['ModelSelection']

class ModelSelection(VerboseLoggingMixin):
    def __init__(self,
                    objective,
                    X,
                    y,
                    sample_weights= None,
                    groups= None,
                    evaluation_weights= None,
                    X_val= None,
                    y_val= None,
                    sample_weights_val= None,
                    groups_val= None,
                    evaluation_weights_val= None,
                    feature_groups= None,
                    reverse= False,
                    preprocessor= None,
                    score_functions= None,
                    validator= None,
                    disable_feature_selection=False,
                    random_state= None,
                    optimizer=SimulatedAnnealing(),
                    cache_dir= None,
                    verbosity= 1):
        self.objective= objective
        self.X= X
        self.y= y
        self.sample_weights= sample_weights
        self.groups= groups
        self.evaluation_weights= evaluation_weights
        self.X_val= X_val
        self.y_val= y_val
        self.sample_weights_val= sample_weights_val
        self.groups_val= groups_val
        self.evaluation_weights_val= evaluation_weights_val
        self.feature_groups= feature_groups
        self.reverse= reverse
        self.preprocessor= preprocessor
        self.score_functions= score_functions
        self.validator= validator
        self.disable_feature_selection=disable_feature_selection
        self.random_state= random_state
        self.cache_dir= cache_dir
        self.optimizer= optimizer
        self.verbosity= verbosity

        VerboseLoggingMixin.__init__(self, "ModelSelection", self.verbosity)
    
    def select(self):
        cache_path= os.path.join(self.cache_dir, self.objective.__name__) if self.cache_dir else None
        self.verbose_logging(1, "Initializing model selection with objective %s" % self.objective.__name__)
        self.verbose_logging(1, "Cache path: %s" % cache_path)

        self.objective_instance= self.objective(self.X, self.y, self.sample_weights, self.groups, self.evaluation_weights,
                                    self.X_val, self.y_val, self.sample_weights_val, self.groups_val,
                                    self.evaluation_weights_val, self.feature_groups, self.reverse,
                                    self.preprocessor, self.score_functions, self.validator, None, self.disable_feature_selection,
                                    self.random_state, cache_path, self.verbosity)

        self.verbose_logging(1, "Executing model selection with objective %s" % self.objective.__name__)
        self.result= self.optimizer.execute(self.objective_instance)

        return self.result

    def get_best_model(self, train=False, X=None, y=None, n_estimators=-1):
        model= self.objective_instance.instantiate(self.result['object'])
        if n_estimators > 1:
            if isinstance(model, ClassifierMixin):
                model= BaggingClassifier(model, n_estimators=n_estimators, bootstrap=False, max_samples=0.9)
            else:
                model= BaggingRegressor(model, n_estimators=n_estimators, bootstrap=False, max_samples=0.9, random_state=self.random_state)
        
        if train is True:
            X, y= X or self.X, y or self.y
            model.fit(X[:,self.result['object']['features']], y)
        
        return {'model': model, 'features': self.result['object']['features'], 'score': self.result['score']}
    
    def get_best_feature_selection_model(self, train=False, X=None, y=None, n_estimators=-1):
        model= self.objective_instance.instantiate(self.result['object']['ml'])
        if n_estimators > 0:
            if isinstance(model, ClassifierMixin):
                model= BaggingClassifier(model, n_estimators=n_estimators, bootstrap=False, max_samples=0.9)
            else:
                model= BaggingRegressor(model, n_estimators=n_estimators, bootstrap=False, max_samples=0.9, random_state=self.random_state)
        
        if isinstance(model, ClassifierMixin):
            model= FeatureSelectionClassifier(model, self.result['object']['features'])
        else:
            model= FeatureSelectionRegressor(model, self.result['object']['features'])
        
        if train is True:
            X, y= X or self.X, y or self.y
            model.fit(X, y)

        return {'model': model, 'features': self.result['object']['features'], 'score': self.result['score']}

    def evaluate(self, 
                    X=None, 
                    y=None, 
                    n_estimators=-1, 
                    validator=None, 
                    score_functions=None,
                    return_vectors=False):
        tmp= self.get_best_model(n_estimators=n_estimators)

        if validator is None:
            if isinstance(tmp['model'], ClassifierMixin):
                validator= RepeatedStratifiedKFold(n_splits=8, n_repeats=3, random_state=self.random_state)
            else:
                validator= RepeatedKFold(n_splits=5, n_repeats=20, random_state=21)
        
        X, y= X or self.X, y or self.y

        if score_functions is None:
            if isinstance(tmp['model'], ClassifierMixin):
                score_functions= [ACC_score()]
            else:
                score_functions= [R2_score()]
        
        score_functions_per_fold= [s.__class__() for s in score_functions]

        model= tmp['model']

        for s in score_functions:
            s.reset()
        for s in score_functions_per_fold:
            s.reset()

        all_tests= []
        all_preds= []
        all_indices= []
        scores_per_fold= [[] for s in score_functions_per_fold]

        i=0
        for train, test in tqdm.tqdm(validator.split(X, y)):
            X_train, X_test= X[train], X[test]
            y_train, y_test= y[train], y[test]
            i=i+1

            model.fit(X_train, y_train)

            y_pred= model.predict(X_test)

            if isinstance(tmp['model'], RegressorMixin):
                if len(y_pred.shape) > 1:
                    y_pred= y_pred[:,0]

            all_tests.append(y_test)
            all_preds.append(y_pred)
            all_indices.append(test)

            for s in score_functions:
                s.accumulate(y_test, y_pred)
            
            for s in score_functions_per_fold:
                s.accumulate(y_test, y_pred)
            
            for i, s in enumerate(score_functions_per_fold):
                scores_per_fold[i].append(s.score())
            
            for s in score_functions_per_fold:
                s.reset()
        print(i)

        return {'scores': [s.score() for s in score_functions],
                'scores_per_fold': scores_per_fold,
                'y_test': np.hstack(all_tests), 
                'y_pred': np.hstack(all_preds), 
                'y_indices': np.hstack(all_indices)}
