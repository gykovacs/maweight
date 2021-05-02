import numpy as np
import pandas as pd
import random
import sys
import glob
import datetime
import os
import os.path
import tqdm

import sklearn.svm as svm
import sklearn.preprocessing as preprocessing
import sklearn.neighbors as neighbors
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

from sklearn.model_selection import KFold, RepeatedKFold

from scipy.stats import spearmanr

from operator import eq

from scipy.stats import skew

import math

from maweight.mltoolkit.base import RandomStateMixin, VerboseLoggingMixin
from ._OptimizationBase import OptimizationBase

__all__=['SimulatedAnnealing']

class SimulatedAnnealing(OptimizationBase, RandomStateMixin, VerboseLoggingMixin):
    def __init__(self,
                max_iterations= 8000,
                T_init= 4,
                annealing_rate= 'auto',
                eps=1e-7,
                maximize= False,
                random_state= None,
                fn_stopping_condition= lambda x: False,
                verbosity= 1,
                logging_frequency= 100):
        self.max_iterations= max_iterations
        self.T_init= T_init
        self.annealing_rate= annealing_rate
        self.eps=eps
        self.maximize= maximize
        self.fn_stopping_condition= fn_stopping_condition
        self.set_random_state(random_state)
        self.verbosity= verbosity
        self.logging_frequency= logging_frequency
        
        if self.annealing_rate == 'auto':
            self.annealing_rate= np.power(self.eps/self.T_init, 1.0/self.max_iterations)
        
        def stopping(x):
            if len(x) < 500:
                return False
            else:
                return np.max(x[-500:])  - np.min(x[-500:]) < 1e-5
        
        self.fn_stopping_condition= stopping

        VerboseLoggingMixin.__init__(self, "SimulatedAnnealing", self.verbosity)
    
    def execute(self, objective):
        self.global_optimum_history= []

        p_space= objective.get_default_parameter_space()

        self.verbose_logging(1, 'creating initial sample')
        T= self.T_init
        s= p_space.sample()
        s_score= objective.score(s)[0]

        if not self.maximize:
            multiplier= 1
        else:
            multiplier= -1

        self.verbose_logging(1, 'executing the main iteration')
        for i in tqdm.tqdm(range(self.max_iterations)):
            T*= self.annealing_rate

            self.verbose_logging(2, 'mutating the object')
            s_new= p_space.mutate(s)

            self.verbose_logging(2, 'scoring the object')
            s_new_score= objective.score(s_new)[0]
            self.verbose_logging(2, 'new_score: %f, best_score: %f' % (s_new_score, s_score))
            
            if multiplier*s_new_score <= multiplier*s_score or (multiplier*s_new_score - multiplier*s_score)/T < math.log(self.random_state.random()):
                s= s_new
                s_score= s_new_score
            
            self.global_optimum_history.append(s_score)

            if self.fn_stopping_condition(self.global_optimum_history):
                print('iterations: %d' % i)
                return {'object': s, 'score': s_score}

            if i % self.logging_frequency == 0:
                self.verbose_logging(1, 'iteration: %d temperature: %.4f current_objective: %.4f' % (i, T, s_score))
                self.verbose_logging(1, 'best_params: %s' % str(s))
        print('iterations: %d' % i)

        self.verbose_logging(1, 'best_params: %s' % str(s))
        self.verbose_logging(1, 'best_score: %s' % str(s_score))
        return {'object': s, 'score': s_score}
