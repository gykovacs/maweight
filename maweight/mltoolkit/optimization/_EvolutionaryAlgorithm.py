#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:02:24 2018

@author: gykovacs
"""

__all__=['EvolutionaryAlgorithm']

import numpy as np
import pandas as pd
import random
import sys
import glob
import datetime
import os
import os.path

from joblib import *

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

import logging

from ._OptimizationBase import *
from maweight.mltoolkit.base import VerboseLoggingMixin

class EvolutionaryAlgorithm(OptimizationBase, VerboseLoggingMixin):
    def __init__(self,
                n_generations= 100,
                max_pop_size= 10,
                n_cross= 10,
                n_mutate= 10,
                mutation_rate= 0.5,
                features_mutation_rate= 1.0,
                max_repeats= 20,
                maximize= False,
                random_state=None,
                fn_stopping_condition= lambda x: False,
                verbosity= 1,
                n_jobs=1):
        self.n_generations= n_generations
        self.max_pop_size= max_pop_size
        self.n_cross= n_cross
        self.n_mutate= n_mutate
        self.mutation_rate= mutation_rate
        self.features_mutation_rate= features_mutation_rate
        self.max_repeats= max_repeats
        self.maximize= maximize
        self.fn_stopping_condition= fn_stopping_condition
        self.verbosity= verbosity
        self.set_verbosity_level(verbosity)
        self.n_jobs= n_jobs
    
    def execute(self, objective):
        self.global_optimum_history= []

        p_space= objective.get_default_parameter_space()

        self.verbose_logging(1, 'creating initial population')
        population= [p_space.sample() for _ in range(2*self.max_pop_size)]
        if self.n_jobs == 1:
            population= [(p, objective.score(p)[0]) for p in population]
        else:
            def scoring(x):
                    import logging
                    logger= logging.getLogger('smote_variants')
                    logger.setLevel(logging.ERROR)
                    return objective.score(x)[0]
            scores= Parallel(n_jobs=self.n_jobs)(delayed(scoring)(p) for p in population)
            population= [(population[i], scores[i]) for i in range(len(population))]

        self.verbose_logging(1, 'running main iteration')
        for i in range(self.n_generations):
            self.verbose_logging(2, 'executing mutations')
            mutations= []
            for _ in range(self.n_mutate):
                mutations.append(p_space.mutate(population[np.random.choice(list(range(len(population))))][0]))

            self.verbose_logging(2, 'executing crossovers')
            crossovers= []
            for _ in range(self.n_cross):
                crossovers.append(p_space.crossover(population[np.random.choice(list(range(len(population))))][0], population[np.random.choice(list(range(len(population))))][0]))
            
            self.verbose_logging(2, 'evaluating mutations (%d) and crossovers (%d)' % (len(mutations), len(crossovers)))

            if self.n_jobs == 1:
                mutations= [(m, objective.score(m)[0]) for m in mutations if objective.validate_parameters(m)]
                crossovers= [(c, objective.score(c)[0]) for c in crossovers if objective.validate_parameters(c)]
            else:
                def scoring(x):
                    import logging
                    logger= logging.getLogger('smote_variants')
                    logger.setLevel(logging.ERROR)
                    return objective.score(x)[0]
                scores= Parallel(n_jobs=self.n_jobs)(delayed(scoring)(m) for m in mutations)
                mutations= [(mutations[i], scores[i]) for i in range(len(mutations))]
                scores= Parallel(n_jobs=self.n_jobs)(delayed(scoring)(c) for c in crossovers)
                crossovers= [(crossovers[i], scores[i]) for i in range(len(crossovers))]

            population.extend(mutations)
            population.extend(crossovers)

            if not self.maximize:
                population= sorted(population, key= lambda x: x[1])
            else:
                population= sorted(population, key= lambda x: -x[1])
            population= population[:self.max_pop_size]

            self.global_optimum_history.append(population[0][1])

            if self.fn_stopping_condition(self.global_optimum_history):
                return population[0][0]

            self.verbose_logging(1, "iteration %d best_score: %f" % (i, population[0][1]))
        
        return {'object': population[0][0], 'score': population[0][1]}


