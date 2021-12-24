#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:02:24 2018

@author: gykovacs
"""

__all__=['KNNR_Objective',
        'SVR_Poly_Objective',
        'SVR_RBF_Objective',
        'XGBR_Objective',
        'LinearRegression_Objective',
        'LassoRegression_Objective',
        'RidgeRegression_Objective',
        'PLSRegression_Objective',
        'RFR_Objective']

import numpy as np

import sklearn.svm as svm
import sklearn.preprocessing as preprocessing
import sklearn.neighbors as neighbors
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn.cross_decomposition as cross_decomposition

import xgboost

from sklearn.linear_model import LinearRegression

from ._ModelSelectionObjective import *
from maweight.mltoolkit.optimization import ParameterSpace, UniformIntegerParameter, CategorialParameter, FixedParameter, JointParameterSpace

random_state= 11

class XGBR_Objective(ModelSelectionObjectiveMixin):
    def __init__(self,
                    X,
                    y,
                    sample_weights=None,
                    groups=None,
                    evaluation_weights=None,
                    X_val=None,
                    y_val=None,
                    sample_weights_val=None,
                    groups_val=None,
                    evaluation_weights_val=None,
                    feature_groups=None,
                    reverse=False,
                    preprocessor=None,
                    score_functions=None,
                    validator=None,
                    oversampler=None,
                    disable_feature_selection=False,
                    random_state=random_state,
                    cache_path=None,
                    verbosity=2):
        super().__init__(xgboost.XGBRegressor, X, y, sample_weights, groups, evaluation_weights,
                            X_val, y_val, sample_weights_val, groups_val, evaluation_weights_val,
                            feature_groups, reverse, preprocessor, score_functions, validator, oversampler, disable_feature_selection, random_state, cache_path, verbosity)
        
        self._default_ml_parameter_space= ParameterSpace({'max_depth': UniformIntegerParameter(2, 5),
                                                            'n_estimators': FixedParameter(100),
                                                            'random_state': FixedParameter(random_state)}, random_state=self._random_state_init)
        self._default_features_parameter_space= super().get_default_parameter_space()
        if not oversampler is None:
            self._oversampler_parameter_space= CategorialParameter(np.random.choice(oversampler.parameter_combinations(), 35))
            self._default_parameter_space= JointParameterSpace({'ml': self._default_ml_parameter_space, 'features': self._default_features_parameter_space, 'oversampler': self._oversampler_parameter_space})
        else:
            self._default_parameter_space= JointParameterSpace({'ml': self._default_ml_parameter_space, 'features': self._default_features_parameter_space})

    def get_default_parameter_space(self):
        """
        Get default parameter descriptors.
        Returns:
            dict: dictionary of default parameter descriptors.
        """
        return self._default_parameter_space

    def instantiate_base(self, parameters):
        return self.base_class(**(parameters['ml']), objective='reg:squarederror')
    
class RFR_Objective(ModelSelectionObjectiveMixin):
    def __init__(self,
                    X,
                    y,
                    sample_weights=None,
                    groups=None,
                    evaluation_weights=None,
                    X_val=None,
                    y_val=None,
                    sample_weights_val=None,
                    groups_val=None,
                    evaluation_weights_val=None,
                    feature_groups=None,
                    reverse=False,
                    preprocessor=None,
                    score_functions=None,
                    validator=None,
                    oversampler=None,
                    disable_feature_selection=False,
                    random_state=random_state,
                    cache_path=None,
                    verbosity=2):
        super().__init__(ensemble.RandomForestRegressor, X, y, sample_weights, groups, evaluation_weights,
                            X_val, y_val, sample_weights_val, groups_val, evaluation_weights_val,
                            feature_groups, reverse, preprocessor, score_functions, validator, oversampler, disable_feature_selection, random_state, cache_path, verbosity)
        
        self._default_ml_parameter_space= ParameterSpace({'max_depth': UniformIntegerParameter(2, 5),
                                                            'n_estimators': FixedParameter(100),
                                                            'random_state': FixedParameter(random_state)}, random_state=self._random_state_init)
        self._default_features_parameter_space= super().get_default_parameter_space()
        if not oversampler is None:
            self._oversampler_parameter_space= CategorialParameter(np.random.choice(oversampler.parameter_combinations(), 35))
            self._default_parameter_space= JointParameterSpace({'ml': self._default_ml_parameter_space, 'features': self._default_features_parameter_space, 'oversampler': self._oversampler_parameter_space})
        else:
            self._default_parameter_space= JointParameterSpace({'ml': self._default_ml_parameter_space, 'features': self._default_features_parameter_space})

    def get_default_parameter_space(self):
        """
        Get default parameter descriptors.
        Returns:
            dict: dictionary of default parameter descriptors.
        """
        return self._default_parameter_space

class KNNR_Objective(ModelSelectionObjectiveMixin):
    def __init__(self, 
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
                    oversampler= None,
                    disable_feature_selection=False,
                    random_state= random_state,
                    cache_path= None,
                    verbosity= 2):
        super().__init__(neighbors.KNeighborsRegressor, X, y, sample_weights, groups, evaluation_weights,
                            X_val, y_val, sample_weights_val, groups_val, evaluation_weights_val,
                            feature_groups, reverse, preprocessor, score_functions, validator, oversampler, disable_feature_selection, random_state, cache_path, verbosity)
        
        self._default_ml_parameter_space= ParameterSpace({'n_neighbors': UniformIntegerParameter(1, 13),
                                                            'weights': CategorialParameter(['uniform', 'distance']),
                                                            'p': UniformIntegerParameter(1, 7)
                                                            }, random_state=self._random_state_init)
        self._default_features_parameter_space= super().get_default_parameter_space()
        if not oversampler is None:
            self._oversampler_parameter_space= CategorialParameter(np.random.choice(oversampler.parameter_combinations(), 35))
            self._default_parameter_space= JointParameterSpace({'ml': self._default_ml_parameter_space, 'features': self._default_features_parameter_space, 'oversampler': self._oversampler_parameter_space})
        else:
            self._default_parameter_space= JointParameterSpace({'ml': self._default_ml_parameter_space, 'features': self._default_features_parameter_space})
    
    def get_default_parameter_space(self):
        """
        Get default parameter descriptors.
        Returns:
            dict: dictionary of default parameter descriptors.
        """
        return self._default_parameter_space

class SVR_Poly_Objective(ModelSelectionObjectiveMixin):
    def __init__(self, 
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
                    preprocessor= preprocessing.StandardScaler(),
                    score_functions= None, 
                    validator= None,
                    oversampler= None,
                    disable_feature_selection=False,
                    random_state= random_state,
                    cache_path= None,
                    verbosity= 2):
        super().__init__(svm.SVR, X, y, sample_weights, groups, evaluation_weights,
                            X_val, y_val, sample_weights_val, groups_val, evaluation_weights_val,
                            feature_groups, reverse, preprocessor, score_functions, validator, oversampler, disable_feature_selection, random_state, cache_path, verbosity)
    
    def get_default_parameter_space(self):
        """
        Get default parameter descriptors.
        Returns:
            dict: dictionary of default parameter descriptors.
        """
        params_ml= ParameterSpace({'degree': UniformIntegerParameter(1, 5),
                                    'C': CategorialParameter([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
                                    'epsilon': CategorialParameter([0.001, 0.01, 0.1, 1.0]),
                                    'kernel': FixedParameter('poly'),
                                    'gamma': FixedParameter('scale')
                                    }, random_state=self._random_state_init)
        params= JointParameterSpace({'ml': params_ml, 'features': super().get_default_parameter_space()})
        
        return params

class SVR_RBF_Objective(ModelSelectionObjectiveMixin):
    def __init__(self, 
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
                    preprocessor= preprocessing.StandardScaler(),
                    score_functions= None, 
                    validator= None,
                    oversampler= None,
                    disable_feature_selection=False,
                    random_state= random_state,
                    cache_path= None,
                    verbosity= 2):
        super().__init__(svm.SVR, X, y, sample_weights, groups, evaluation_weights,
                            X_val, y_val, sample_weights_val, groups_val, evaluation_weights_val,
                            feature_groups, reverse, preprocessor, score_functions, validator, oversampler, disable_feature_selection, random_state, cache_path, verbosity)
    
    def get_default_parameter_space(self):
        """
        Get default parameter descriptors.
        Returns:
            dict: dictionary of default parameter descriptors.
        """
        params_ml= ParameterSpace({ 'C': CategorialParameter([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
                                    'epsilon': CategorialParameter([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]),
                                    'kernel': FixedParameter('rbf'),
                                    'gamma': FixedParameter('scale')
                                    }, random_state=self._random_state_init)
        params= JointParameterSpace({'ml': params_ml, 'features': super().get_default_parameter_space()})
        
        return params

class LinearRegression_Objective(ModelSelectionObjectiveMixin):
    def __init__(self,
                    X,
                    y,
                    sample_weights=None,
                    groups=None,
                    evaluation_weights=None,
                    X_val=None,
                    y_val=None,
                    sample_weights_val=None,
                    groups_val=None,
                    evaluation_weights_val=None,
                    feature_groups=None,
                    reverse=False,
                    preprocessor=preprocessing.StandardScaler(),
                    score_functions=None,
                    validator=None,
                    oversampler=None,
                    disable_feature_selection=False,
                    random_state=random_state,
                    cache_path=None,
                    verbosity=2):
        super().__init__(linear_model.LinearRegression, X, y, sample_weights, groups, evaluation_weights,
                        X_val, y_val, sample_weights_val, groups_val, evaluation_weights_val,
                        feature_groups, reverse, preprocessor, score_functions, validator, oversampler, 
                        disable_feature_selection, random_state, cache_path, verbosity)
    
    def get_default_parameter_space(self):
        params_ml= ParameterSpace({'fit_intercept': UniformIntegerParameter(0, 1, random_state=self._random_state_init)}, random_state=self._random_state_init)
        params= JointParameterSpace({'ml': params_ml, 'features': super().get_default_parameter_space()})
        return params

class LassoRegression_Objective(ModelSelectionObjectiveMixin):
    def __init__(self,
                    X,
                    y,
                    sample_weights=None,
                    groups=None,
                    evaluation_weights=None,
                    X_val=None,
                    y_val=None,
                    sample_weights_val=None,
                    groups_val=None,
                    evaluation_weights_val=None,
                    feature_groups=None,
                    reverse=False,
                    preprocessor=preprocessing.StandardScaler(),
                    score_functions=None,
                    validator=None,
                    oversampler=None,
                    disable_feature_selection=False,
                    random_state=random_state,
                    cache_path=None,
                    verbosity=2):
        super().__init__(linear_model.Lasso, X, y, sample_weights, groups, evaluation_weights,
                        X_val, y_val, sample_weights_val, groups_val, evaluation_weights_val,
                        feature_groups, reverse, preprocessor, score_functions, validator, oversampler, 
                        disable_feature_selection, random_state, cache_path, verbosity)
    
    def get_default_parameter_space(self):
        params_ml= ParameterSpace({'fit_intercept': FixedParameter(True), #UniformIntegerParameter(0, 1, random_state=self._random_state_init),
                                   'alpha': CategorialParameter([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
                                   'random_state': FixedParameter(5)}, random_state=self._random_state_init)
        params= JointParameterSpace({'ml': params_ml, 'features': super().get_default_parameter_space()})
        return params

class RidgeRegression_Objective(ModelSelectionObjectiveMixin):
    def __init__(self,
                    X,
                    y,
                    sample_weights=None,
                    groups=None,
                    evaluation_weights=None,
                    X_val=None,
                    y_val=None,
                    sample_weights_val=None,
                    groups_val=None,
                    evaluation_weights_val=None,
                    feature_groups=None,
                    reverse=False,
                    preprocessor=preprocessing.StandardScaler(),
                    score_functions=None,
                    validator=None,
                    oversampler=None,
                    disable_feature_selection=False,
                    random_state=random_state,
                    cache_path=None,
                    verbosity=2):
        super().__init__(linear_model.Ridge, X, y, sample_weights, groups, evaluation_weights,
                        X_val, y_val, sample_weights_val, groups_val, evaluation_weights_val,
                        feature_groups, reverse, preprocessor, score_functions, validator, oversampler, 
                        disable_feature_selection, random_state, cache_path, verbosity)
    
    def get_default_parameter_space(self):
        params_ml= ParameterSpace({'fit_intercept': FixedParameter(True), #UniformIntegerParameter(0, 1, random_state=self._random_state_init),
                                   'alpha': CategorialParameter([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
                                   'random_state': FixedParameter(5)}, random_state=self._random_state_init)
        params= JointParameterSpace({'ml': params_ml, 'features': super().get_default_parameter_space()})
        return params

class PLSRegression_Objective(ModelSelectionObjectiveMixin):
    def __init__(self,
                    X,
                    y,
                    sample_weights=None,
                    groups=None,
                    evaluation_weights=None,
                    X_val=None,
                    y_val=None,
                    sample_weights_val=None,
                    groups_val=None,
                    evaluation_weights_val=None,
                    feature_groups=None,
                    reverse=False,
                    preprocessor=preprocessing.StandardScaler(),
                    score_functions=None,
                    validator=None,
                    oversampler=None,
                    disable_feature_selection=False,
                    random_state=random_state,
                    cache_path=None,
                    verbosity=2):
        super().__init__(cross_decomposition.PLSRegression, X, y, sample_weights, groups, evaluation_weights,
                        X_val, y_val, sample_weights_val, groups_val, evaluation_weights_val,
                        feature_groups, reverse, preprocessor, score_functions, validator, oversampler, disable_feature_selection, random_state, cache_path, verbosity)
    
    def get_default_parameter_space(self):
        max_components= int(np.sqrt(len(self.X[0])))
        params_ml= ParameterSpace({'n_components': UniformIntegerParameter(2, max_components, random_state=self._random_state_init)}, random_state=self._random_state_init)
        params= JointParameterSpace({'ml': params_ml, 'features': super().get_default_parameter_space()})
        return params
