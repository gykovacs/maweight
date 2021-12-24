
__all__=['ModelSelectionObjectiveMixin',
         'ModelSelectionRegressor',
         'FeatureSelectionRegressor',
         'RMSE_score',
         'RMSLE_score',
         'MAE_score',
         'R2_score',
         'NegR2_score']

import numpy as np
import json
import tqdm

from sklearn.metrics import confusion_matrix, roc_auc_score
import sklearn.preprocessing as preprocessing
import sklearn.metrics as metrics
import sklearn.ensemble as ensemble
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
import sklearn.base
import scipy.stats
from sklearn.base import ClassifierMixin, RegressorMixin

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from maweight.mltoolkit.base import CacheBase, VerboseLoggingMixin, RandomStateMixin
from maweight.mltoolkit.optimization import *

from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler

import copy

random_state= 11

class RMSLE_score:
    def __init__(self, mode= 'max', threshold= None):
        self.mode= mode
        self.threshold= threshold
        self.reset()
    
    def reset(self):
        self.sum_of_squares= 0.0
        self.n= 0
        self.groups= None
    
    def accumulate(self, gt, pred, groups= None):
        if groups is None:
            self.sum_of_squares= self.sum_of_squares + np.sum(np.power(np.log(np.exp(gt) + 1.0) - np.log(np.exp(pred) + 1.0), 2.0))
            self.n= self.n + len(gt)
        else:
            results= pd.DataFrame({'gt': gt, 'pred': pred, 'groups': groups})
            results['squared_diff']= (np.log(np.exp(results['gt']) + 1.0) - np.log(np.exp(results['pred']) + 1.0)).apply(lambda x: x*x)
            results['n']= 1
            grouped= results.groupby(by='groups').agg({'squared_diff': lambda x: np.sum(x), 'n': lambda x: np.sum(x)})
            if self.groups is None:
                self.groups= grouped
            else:
                self.groups= self.groups.add(grouped, fill_value= 0.0)
                
        return self.score()
    
    def score(self):
        if self.groups is None:
            return np.sqrt(self.sum_of_squares/self.n)
        else:
            scores= (self.groups['squared_diff']/self.groups['n']).apply(lambda x: np.sqrt(x))
            if self.mode == 'max':
                return np.max(scores.values)
            elif self.mode == 'min':
                return np.min(scores.values)
            elif self.mode == 'mean':
                return np.mean(scores.values)
            elif self.mode == 'fraction_below':
                return np.sum(scores < self.threshold)/len(scores)
            elif self.mode == 'fraction_above':
                return np.sum(scores > self.threshold)/len(scores)

class MAE_score:
    def __init__(self, mode= 'max', threshold= None):
        self.mode= mode
        self.threshold= threshold
        self.reset()
        
    def reset(self):
        self.sum_of_errors= 0.0
        self.n= 0
        self.groups= None
        
    def accumulate(self, gt, pred, groups= None):
        if groups is None:
            self.sum_of_errors= self.sum_of_errors + np.sum(np.abs(gt - pred))
            self.n= self.n + len(gt)
        else:
            results= pd.DataFrame({'gt': gt, 'pred': pred, 'groups': groups})
            results['sum_diff']= np.abs(results['gt'] - results['pred'])
            results['n']= 1
            grouped= results.groupby(by='groups').agg({'sum_diff': lambda x: np.sum(x), 'n': lambda x: np.sum(x)})
            if self.groups is None:
                self.groups= grouped
            else:
                self.groups= self.groups.add(grouped, fill_value= 0.0)
                
        return self.score()
    
        
    def score(self):
        if self.groups is None:
            return self.sum_of_errors/self.n
        else:
            scores= self.groups['sum_diff']/self.groups['n']
            if self.mode == 'max':
                return np.max(scores.values)
            elif self.mode == 'min':
                return np.min(scores.values)
            elif self.mode == 'mean':
                return np.mean(scores.values)
            elif self.mode == 'fraction_below':
                return np.sum(scores < self.threshold)/len(scores)
            elif self.mode == 'fraction_above':
                return np.sum(scores > self.threshold)/len(scores)

class GroupScoreMixin:
    def evaluate_scores(self, group_mode, scores, threshold):
        if self.group_mode == 'max':
            return -np.max(list(scores.values()))
        if self.group_mode == 'min':
            return -np.min(list(scores.values()))
        if self.group_mode == 'mean':
            return -np.mean(list(scores.values()))
        if self.group_mode == 'fraction_below':
            return np.sum(np.array(scores.values()) < self.threshold)/len(scores)
        if self.group_mode == 'fraction_below':
            return -np.sum(np.array(scores.values()) > self.threshold)/len(scores)

class R2_score(GroupScoreMixin):
    def __init__(self, group_mode= 'max', threshold= None):
        self.group_mode= group_mode
        self.threshold= threshold
        self.reset()
        
    def reset(self):
        self.sum_of_squares= 0.0
        self.sum_of_values= 0.0
        self.sum_of_residual_squares= 0.0
        self.group_sum_of_squares= {}
        self.group_sum_of_values= {}
        self.group_sum_of_residual_squares= {}
        self.n= 0
        self.group_n= {}
        
    def accumulate(self, gt, pred, groups= None, sample_weights= None):
        if groups is None:
            self.sum_of_squares+= np.dot(gt, gt)
            self.sum_of_values+= np.sum(gt)
            self.sum_of_residual_squares+= np.dot((gt - pred), (gt - pred))
            self.n+= len(gt)
        else:
            for g in groups:
                if not g in self.group_sum_of_squares:
                    self.group_sum_of_squares[g]= 0
                    self.group_sum_of_values[g]= 0
                    self.group_sum_of_residual_squares[g]= 0
                    self.group_n[g]= 0
                self.group_sum_of_squares[g]+= np.dot(gt[groups[g]], gt[groups[g]])
                self.group_sum_of_values[g]+= np.sum(gt[groups[g]])
                self.group_sum_of_residual_squares[g]+= np.dot((gt[groups[g]] - pred[groups[g]]), (gt[groups[g]] - pred[groups[g]]))
                self.group_n[g]+= len(groups[g])

        return self.score()
        
    def score(self):
        if len(self.group_n) == 0:
            return (1.0 - self.sum_of_residual_squares / (self.sum_of_squares - self.sum_of_values*self.sum_of_values/self.n))
        else:
            scores= {}
            for g in self.group_n:
                scores[g]= (1.0 - self.group_sum_of_residual_squares[g]/(self.group_sum_of_squares[g] - self.group_sum_of_values[g]*self.group_sum_of_values[g]/self.group_n[g]))

            return self.evaluate_scores(self.group_mode, scores, self.threshold)

class RMSE_score(GroupScoreMixin):
    def __init__(self, group_mode= 'max', threshold= None):
        self.group_mode= group_mode
        self.threshold= threshold
        self.reset()
    
    def reset(self):
        self.sum_of_residual_squares= 0.0
        self.group_sum_of_residual_squares= {}
        self.n= 0
        self.group_n= {}
    
    def accumulate(self, gt, pred, groups= None, sample_weights= None):
        if groups is None:
            self.sum_of_residual_squares+= np.dot(gt - pred, gt - pred)
            self.n+= len(gt)
        else:
            for g in groups:
                if not g in self.group_sum_of_residual_squares:
                    self.group_sum_of_residual_squares[g]= 0
                    self.group_n[g]= 0
                self.group_sum_of_residual_squares[g]+= np.dot(gt[groups[g]] - pred[groups[g]], gt[groups[g]] - pred[groups[g]])
                self.group_n[g]+= len(groups[g])
                
        return self.score()
    
    def score(self):
        if len(self.group_n) == 0:
            return np.sqrt(self.sum_of_residual_squares/self.n)
        else:
            scores= {}
            for g in self.group_n:
                scores[g]= np.sqrt(self.group_sum_of_residual_squares[g]/self.group_n[g])
            return self.evaluate_scores(self.group_mode, scores, self.threshold)            

class NegR2_score(R2_score):
    def __init__(self, group_mode= 'max', threshold= None):
        super().__init__(group_mode, threshold)
        
    def accumulate(self, gt, pred, groups= None, sample_weights= None):
        return -super().accumulate(gt, pred, groups, sample_weights)
        
    def score(self):
        return -super().score()

class FeatureSelectionRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                    regressor,
                    feature_mask):
        self.regressor=regressor
        self.feature_mask=feature_mask
    
    def fit(self, X, y, sample_weights=None):
        self.ss= StandardScaler()
        X_tmp= self.ss.fit_transform(X[:,self.feature_mask])
        try:
            self.regressor.fit(X_tmp, y, sample_weights=sample_weights)
        except:
            self.regressor.fit(X_tmp, y)
        return self
    
    def predict(self, X):
        return self.regressor.predict(self.ss.transform(X[:,self.feature_mask]))

class ModelSelectionRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                    objective, 
                    optimizer,
                    bagging_params= {},
                    random_state=None):
        self.objective= objective
        self.optimizer= optimizer
        self.bagging_params= bagging_params
        self.is_frozen= False
        self.random_state= random_state
    
    def freeze(self):
        self.is_frozen= True

    def fit(self, X, y):
        self.objective.X= X
        self.objective.y= y

        if not self.is_frozen:
            self.best_parameters= self.optimizer.execute(self.objective)
        
        #logging.info('best_parameters: %s' % str(self.best_parameters))

        if self.bagging_params == {}:
            self.model= self.objective.instantiate(self.best_parameters)
        else:
            self.model= BaggingRegressor(base_estimator=self.objective.instantiate(self.best_parameters), **(self.bagging_params), random_state=self.random_state)
        
        self.model.fit(X[:,self.best_parameters['features']], y)

        return self
    
    def predict(self, X):
        return self.model.predict(X[:,self.best_parameters['features']])
    
    def predict_std(self, X):
        if isinstance(self.model, BaggingRegressor):
            ens_preds= []
            for e in self.model.estimators_:
                ens_preds.append(e.predict(X[:,self.best_parameters['features']]))
            ens_preds= np.stack(ens_preds, axis=1)
            return self.model.predict(X[:,self.best_parameters['features']]), np.std(ens_preds, axis=1)
        else:
            return self.model.predict(X[:,self.best_parameters['features']]), np.repeat(0.0, len(X))
    
    def score(self):
        return self.best_parameters['score']

class ModelSelectionObjectiveMixin(RandomStateMixin, CacheBase, VerboseLoggingMixin):
    """
    Represents a general classification or regression model.
    """
    def __init__(self,
                 base_class,
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
                 validator=None,
                 oversampler=None,
                 disable_feature_selection=False,
                 random_state= None,
                 cache_path= None,
                 verbosity= 2):
        """
        Constructor of the Model class
        Args:
            feature_names (list(str)): names of the features to use
        """
        self.base_class= base_class
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
        self.oversampler= oversampler
        self.disable_feature_selection=disable_feature_selection
        self.set_random_state(random_state)
        self.verbosity= verbosity

        CacheBase.__init__(self, cache_path)
        VerboseLoggingMixin.__init__(self, "ModelSelectionObjective", self.verbosity)
    
    def validate_parameters(self, parameters):
        """
        Checks the consistency of the actual parameters.
        Returns:
            bool: true if the parameters are consistent
        """
        return True

    def get_default_parameter_space(self):
        if self.feature_groups is None:
            n_init= 1 if not self.reverse else len(self.X[0])
            return BinaryVectorParameter(len(self.X[0]), n_init=n_init, random_state=self._random_state_init, disabled=self.disable_feature_selection)
        else:
            n_init= 1 if not self.reverse else len(self.feature_groups)
            return GroupedBinaryVectorParameter(length= len(self.X[0]), groups= self.feature_groups, n_init=n_init, random_state=self._random_state_init)

    def _score_ab(self, parameters, model, X, y, sample_weights, X_val, y_val, groups_val, evaluation_weights_val, score_functions):
        # trying to fit with sample_weights
        try:
            model.fit(X, y, sample_weights=sample_weights)
        except:
            model.fit(X, y)

        if issubclass(self.base_class, ClassifierMixin) or issubclass(self.base_class, XGBClassifier):
            # if the problem is classification
            y_pred, y_pred_proba= None, None
            # trying to predict probabilities
            try:
                y_pred_proba= model.predict_proba(X_val)
            except:
                y_pred= model.predict(X_val)
            
            # predicting labels from probabilities
            if y_pred is None:
                y_pred= np.apply_along_axis(lambda x: np.argmax(x), axis=1, arr=y_pred_proba)
            
            # scoring
            for s in score_functions:
                s.accumulate(y_val, y_pred, y_pred_proba, groups_val, evaluation_weights_val)
        elif issubclass(self.base_class, RegressorMixin) or issubclass(self.base_class, XGBRegressor):
            # if the problem is regression, prediction
            y_pred= model.predict(X_val)

            # scoring
            for s in score_functions:
                s.accumulate(y_val, y_pred, groups_val, evaluation_weights_val)

    def _score_with_validator(self, parameters, model, validator=None, score_functions=None):
        validator= validator or self.validator

        for train, test in validator.split(self.X, self.y):
            X_train, X_test= self.X[train], self.X[test]
            y_train, y_test= self.y[train], self.y[test]

            # if groups and sample weights are provided, spliting is applied
            groups_test, sample_weights_train, sample_weights_test, evaluation_weights_train, evaluation_weights_test= None, None, None, None, None
            if self.groups:
                groups_test= self.groups[test]
            if self.sample_weights:
                sample_weights_train, sample_weights_test= self.sample_weights[train], self.sample_weights[test]
                evaluation_weights_train, evaluation_weights_test= self.evaluation_weights[train], self.evaluation_weights[test]
            
            # trying to fit the model with sample weights
            try:
                model.fit(X_train, y_train, sample_weights=sample_weights_train)
            except:
                model.fit(X_train, y_train)

            if issubclass(self.base_class, ClassifierMixin) or issubclass(self.base_class, XGBClassifier):
                # if the problem is classification
                y_pred, y_pred_proba= None, None
                # trying to predict probabilities
                try:
                    y_pred_proba= model.predict_proba(X_test)
                except Exception as e:
                    print(e)
                    y_pred= model.predict(X_test)
                
                # predicting labels from probabilities
                if y_pred is None:
                    y_pred= np.apply_along_axis(lambda x: np.argmax(x), axis=1, arr=y_pred_proba)

                # scoring
                for s in score_functions:
                    s.accumulate(y_test, y_pred, y_pred_proba, groups_test, evaluation_weights_test)
            elif issubclass(self.base_class, RegressorMixin) or issubclass(self.base_class, XGBRegressor):
                # if the problem is regression, prediction
                y_pred= model.predict(X_test)
                #print('y_pred.shape', y_pred.shape)
                if len(y_pred.shape) > 1:
                    y_pred= y_pred[:,0]
                #print('mod_y_pred.shape', y_pred.shape)

                # scoring
                for s in score_functions:
                    s.accumulate(y_test, y_pred, groups_test, evaluation_weights_test)

    def score(self, 
                parameters, 
                X= None, 
                y= None, 
                groups=None, 
                evaluation_weights=None,
                validator=None):
        """
        Validate the model.
        Args:
            X (pd.DataFrame): validation training data
            y (pd.DataFrame): validation test data
            score_function (func): the score function to use
            bagging (int): the number of random repetitions of k-fold cross validators
        Returns:
            float, np.array: the validation score and the mean forecasts on the training set
        """

        self.verbose_logging(2, "scoring %s" % str(parameters))
        
        parameters_string= ParameterSpace.jsonify(parameters)

        if self.is_in_cache(parameters_string):
            self.verbose_logging(1, 'no evaluation, score taken from cache')
            return self.get_from_cache(parameters_string)

        if self.score_functions is None and issubclass(self.base_class, sklearn.base.ClassifierMixin):
            self.verbose_logging(2, "the score function to be used is ACC_score")
            self.score_functions= [NegACC_score()]
        elif self.score_functions is None:
            self.verbose_logging(2, "the score function to be used is R2_score")
            self.score_functions= [NegR2_score()]
        
        score_functions= copy.deepcopy(self.score_functions)

        model= self.instantiate(parameters)

        validator= validator or copy.deepcopy(self.validator)

        if validator is None:
            if issubclass(self.base_class, sklearn.base.ClassifierMixin) or issubclass(self.base_class, XGBClassifier):
                validator= RepeatedStratifiedKFold(n_splits= 8, n_repeats= 3, random_state=5)
            else:
                validator= RepeatedKFold(n_splits= 5, n_repeats= 20, random_state=random_state)
            self.verbose_logging(2, "validator created %s" % str(validator))
        
        self.verbose_logging(2, "resetting score functions")
        for s in score_functions:
            s.reset()
        
        # cross-validation based evaluation using the preset vectors
        if ((X is None and self.X_val is None) or self.X == X):
            # evaluation goes by the preset vectors
            self._score_with_validator(parameters, model, validator, score_functions)
        elif not X is None:
            # if we are validating on new data, the model is fitted to the preset dataset
            self._score_ab(parameters, model, self.X, self.y, self.sample_weights, X, y, groups, evaluation_weights, score_functions)
        elif not self.X_val is None:
            # validation happens on the validation dataset
            self._score_ab(parameters, model, self.X, self.y, self.sample_weights, self.X_val, self.y_val, self.groups_val, self.evaluation_weights_val, score_functions)

        scores= [s.score() for s in score_functions]

        self.put_into_cache(parameters_string, scores)

        return scores
    
    def instantiate_base(self, parameters):
        if 'n_components' in parameters['ml'] and parameters['ml']['n_components'] > np.sum(parameters['features']):
            parameters['ml']['n_components']= np.sum(parameters['features'])
        from sklearn import linear_model
        from sklearn import pipeline
        if (not self.base_class == linear_model.Lasso) and (not self.base_class == linear_model.Ridge):
            return self.base_class(**(parameters['ml']))
        elif self.base_class == linear_model.Lasso:
            return pipeline.make_pipeline(StandardScaler(with_mean=False), linear_model.Lasso(**(parameters['ml'])))
        elif self.base_class == linear_model.Ridge:
            return pipeline.make_pipeline(StandardScaler(with_mean=False), linear_model.Ridge(**(parameters['ml'])))

    def instantiate(self, parameters, features_to_ignore=None):
        """
        Instantiate new model instance.
        Args:
            parameters (dict): dictionary of parameters
        Returns:
            RegressorMixin: a new RegressorMixin object
        """
        
        if self.preprocessor:
            if not self.oversampler:
                model= Pipeline(steps=[('prep', copy.deepcopy(self.preprocessor)), ('model', self.instantiate_base(parameters))])
            else:
                model= Pipeline(steps=[('prep', copy.deepcopy(self.preprocessor)), ('model', sv.OversamplingClassifier(self.oversampler(**parameters['oversampler']), self.instantiate_base(parameters)))])
        else:
            model= self.instantiate_base(parameters)
        
        if issubclass(self.base_class, ClassifierMixin):
            if not features_to_ignore:
                model= FeatureSelectionClassifier(model, parameters['features'])
            else:
                features= np.delete(parameters['features'], features_to_ignore)
                print('features reduced from %d to %d by removing %s' % (len(parameters['features']), len(features), str(features_to_ignore)))
                model= FeatureSelectionClassifier(model, features)
        else:
            if not features_to_ignore:
                model= FeatureSelectionRegressor(model, parameters['features'])
            else:
                features= np.delete(parameters['features'], features_to_ignore)
                print('features reduced from %d to %d by removing %s' % (len(parameters['features']), len(features), str(features_to_ignore)))
                model= FeatureSelectionRegressor(model, features)

        return model



