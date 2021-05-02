import numpy as np
import copy
import json

from maweight.mltoolkit.base import RandomStateMixin

class ParameterBase(RandomStateMixin):
    pass

class FixedParameter(ParameterBase):
    def __init__(self, value):
        self.value= value
    
    def sample(self):
        return self.value
    
    def dim(self):
        return 1
    
    def vectorize(self, parameter):
        return np.array(parameter)
    
    def devectorize(self, vector):
        return vector[0]
    
    def mutate(self, parameter, rate):
        return self.sample()
    
    def ranges(self):
        return np.array([self.value, self.value])

    def cap(self, value):
        return self.value

class UniformFloatParameter(ParameterBase):
    def __init__(self, lower_bound, upper_bound, random_state= None):
        self.lower_bound= lower_bound
        self.upper_bound= upper_bound
        self.set_random_state(random_state)
    
    def sample(self):
        return self.lower_bound + self.random_state.random()*(self.upper_bound - self.lower_bound)
    
    def dim(self):
        return 1
    
    def vectorize(self, parameter):
        return np.array(self.cap(parameter))
    
    def devectorize(self, vector):
        return self.cap(vector[0])

    def mutate(self, parameter, rate= 0.5):
        return self.sample()
    
    def ranges(self):
        return np.array([self.lower_bound, self.upper_bound])
    
    def cap(self, value):
        if value < self.lower_bound:
            return self.lower_bound
        elif value > self.upper_bound:
            return self.upper_bound
        return value

class UniformIntegerParameter(ParameterBase):
    def __init__(self, lower_bound, upper_bound, random_state= None):
        self.lower_bound= lower_bound
        self.upper_bound= upper_bound
        
        self.set_random_state(random_state)
    
    def sample(self):
        return self.random_state.randint(self.lower_bound, self.upper_bound + 1)
    
    def dim(self):
        return 1
    
    def vectorize(self, parameter):
        return np.array(int(round(self.cap(parameter))))
    
    def devectorize(self, vector):
        return self.cap(int(round(vector[0])))

    def mutate(self, parameter, rate= 0.5):
        return self.sample()
    
    def ranges(self):
        return np.array([self.lower_bound, self.upper_bound])
    
    def cap(self, value):
        if value < self.lower_bound:
            return self.lower_bound
        if value > self.upper_bound:
            return self.upper_bound
        return value

class CategorialParameter(ParameterBase):
    def __init__(self, categories, random_state=None):
        self.categories= categories
        self.map_to_vector= {str(c): np.eye(len(categories))[i] for i, c in enumerate(categories)}
        self.set_random_state(random_state)
    
    def sample(self):
        return self.random_state.choice(self.categories)
    
    def dim(self):
        return len(self.categories)
    
    def vectorize(self, parameter):
        return self.map_to_vector[str(parameter)]
    
    def devectorize(self, vector):
        return self.categories[np.argmin(np.apply_along_axis(lambda x: np.linalg.norm(vector - x), axis=0, arr=np.eye(len(self.categories))))]
    
    def mutate(self, parameter, rate=0.1):
        return self.sample()
    
    def ranges(self):
        return np.repeat([[0, 1]], len(self.categories), axis=0)

    def cap(self, vector):
        return self.vectorize(self.devectorize(vector))
    
    def crossover(self, parameters1, parameters2, crossover_rate= 0.5):
        new_parameters= self.sample()
        for p in parameters1:
            if self.random_state.random() < 0.5:
                new_parameters[p]= parameters1[p]
            else:
                new_parameters[p]= parameters2[p]
        return new_parameters

class ParameterSpace(RandomStateMixin):
    def __init__(self, parameters, random_state= None):
        self.parameters= parameters
        self.set_random_state(random_state)
        self.init_inner_structure()
    
    def init_inner_structure(self):
        self.dimensions= {p: self.parameters[p].dim() for p in self.parameters}
        if not self.random_state is None:
            for p in self.parameters:
                self.parameters[p].set_random_state(self.random_state)

    def dim(self):
        return np.sum(list(self.dimensions.values()))

    def sample(self):
        result= {}
        for p in self.parameters:
            result[p]= self.parameters[p].sample()
        return result
    
    def sample_vectorized(self):
        return self.vectorize(self.sample())

    def vectorize(self, parameters):
        return np.hstack([self.parameters[p].vectorize(parameters[p]) for p in parameters])
    
    def devectorize(self, vector):
        result= {}
        n= 0
        for p in self.parameters:
            result[p]= self.parameters[p].devectorize(vector[n:(n+self.dimensions[p])])
            n+= self.dimensions[p]
        
        return result
    
    def mutate(self, parameters, rate= 0.5):
        new_parameters= copy.deepcopy(parameters)
        for p in self.parameters:
            if self.random_state.random() < rate:
                new_parameters[p]= self.parameters[p].mutate(new_parameters[p], rate)
        
        return new_parameters
    
    def crossover(self, parameters1, parameters2, crossover_rate= 0.5):
        new_parameters= self.sample()
        for p in parameters1:
            if self.random_state.random() < crossover_rate:
                if self.random_state.random() < 0.5:
                    new_parameters[p]= parameters1[p]
                else:
                    new_parameters[p]= parameters2[p]
        return new_parameters
    
    def ranges(self):
        return np.vstack([self.parameters[p].ranges() for p in self.parameters])
    
    def cap(self, vector):
        return self.vectorize(self.devectorize(vector))
    
    @classmethod
    def encode_ndarrays(cls, parameters):
        for p in parameters:
            if isinstance(parameters[p], np.ndarray):
                parameters[p]= parameters[p].tolist()
            elif isinstance(parameters[p], dict):
                parameters[p]= ParameterSpace.encode_ndarrays(parameters[p])
            elif isinstance(parameters[p], np.int64):
                parameters[p]= int(parameters[p])
        return parameters

    @classmethod
    def decode_ndarrays(cls, parameters):
        for p in parameters:
            if isinstance(parameters[p], list):
                parameters[p]= np.array(parameters[p])
            elif isinstance(parameters[p], dict):
                parameters[p]= ParameterSpace.decode_ndarrays(parameters[p])
        return parameters

    @classmethod
    def jsonify(cls, parameters):
        parameters= copy.deepcopy(parameters)
        return json.dumps(ParameterSpace.encode_ndarrays(parameters))
    
    @classmethod
    def dejsonify(cls, parameters):
        return ParameterSpace.decode_ndarrays(json.loads(parameters))

class BinaryVectorParameter(ParameterBase, ParameterSpace):
    def __init__(self, n, n_lower=1, n_upper=None, n_init=None, random_state=None, disabled=False):
        self.n= n
        self.n_lower= n_lower
        self.n_upper= n_upper
        self.n_init= n_init
        self.disabled= disabled
        
        ParameterBase.__init__(self)
        ParameterSpace.__init__(self, {'binary_vector': self}, random_state)
    
    def sample(self):
        if not self.disabled:
            tmp= np.repeat(False, self.n)
            tmp[self.random_state.choice(list(range(self.n)), self.n_init or int(self.n/2), replace=False)]= True
        else:
            tmp= np.repeat(True, self.n)
        
        return tmp
    
    def dim(self):
        return self.n
    
    def vectorize(self, parameter):
        return parameter.astype(float)
    
    def devectorize(self, vector):
        sort= sorted(vector)

        if self.n_lower is None and self.n_upper is None:
            tmp= vector >= 0.5
            if sum(tmp) == 0:
                tmp= np.repeat(False, self.n)
                tmp[self.random_state.choice(list(range(self.n)), self.n_init or int(self.n/2), replace=False)]= True
            return tmp
        else:
            n_lower= self.n_lower or 0
            n_upper= self.n_upper or self.n

            binary_vector= vector >= 0.5
            if sum(binary_vector) >= n_lower and sum(binary_vector) <= n_upper:
                return binary_vector
            else:
                n= int((n_lower + n_upper)/2)
            tmp= vector >= sort[n]
            if sum(tmp) == 0:
                tmp= np.repeat(False, self.n)
                tmp[self.random_state.choice(list(range(self.n)), self.n_init or int(self.n/2), replace=False)]= True
            return tmp
    
    def mutate(self, parameter, rate= None):
        if self.disabled:
            return parameter
        
        new_parameter= parameter.copy()
        current_n= np.sum(new_parameter)

        if rate == None:
            increase, decrease= False, False

            if (self.n_lower is None and current_n > 0) or (not self.n_lower is None and current_n > self.n_lower):
                decrease= True
            if (self.n_upper is None and current_n < self.n - 1) or (not self.n_upper is None and current_n < self.n_upper):
                increase= True
            
            if decrease and increase:
                idx= self.random_state.choice(list(range(self.n)))
            elif decrease:
                idx= self.random_state.choice(np.where(new_parameter == True)[0])
            elif increase:
                idx= self.random_state.choice(np.where(new_parameter == False)[0])
            
            new_parameter[idx]= not new_parameter[idx]
        else:
            for i in range(len(new_parameter)):
                if self.random_state.random() < rate:
                    new_parameter[i]= not new_parameter[i]
            if not self.n_lower is None and sum(new_parameter) < self.n_lower:
                new_parameter[self.random_state.choice(np.where(new_parameter == False)[0], self.n_lower - sum(new_parameter), replace=False)]= True
            elif not self.n_upper is None and sum(new_parameter) > self.n_upper:
                new_parameter[self.random_state.choice(np.where(new_parameter == True)[0], sum(new_parameter) - self.n_upper, replace=False)]= False

        tmp= new_parameter
        if sum(tmp) == 0:
            tmp= np.repeat(False, self.n)
            tmp[self.random_state.choice(list(range(self.n)), self.n_init or int(self.n/2), replace=False)]= True

        return tmp
    
    def crossover(self, parameter1, parameter2, crossover_rate= 0.9):
        if self.disabled:
            return parameter1
        new_parameter= parameter1.copy()

        for i in range(len(parameter2)):
            if crossover_rate is None or self.random_state.random() < crossover_rate:
                new_parameter[i]= parameter2[i] if self.random_state.randint(2) == 1 else parameter1[i]
        
        current_n= np.sum(new_parameter)

        increase, decrease= 0, 0
        if not self.n_lower is None and current_n < self.n_lower:
            increase= self.n_lower - current_n
        if not self.n_upper is None and current_n > self.n_upper:
            decrease= current_n - self.n_upper
        
        if increase > 0:
            already_set= set(np.where(new_parameter == True)[0])
            set_1= set(np.where(parameter1 == True)[0])
            set_2= set(np.where(parameter2 == True)[0])
            not_set_1= set_1.difference(already_set)
            not_set_2= set_2.difference(already_set)
            new_parameter[self.random_state.choice(list(not_set_1.union(not_set_2)), increase)]= True
        if decrease > 0:
            already_set= np.where(new_parameter == True)[0]
            new_parameter[self.random_state.choice(already_set, decrease)]= False

        tmp= new_parameter
        if sum(tmp) == 0:
            tmp= np.repeat(False, self.n)
            tmp[self.random_state.choice(list(range(self.n)), self.n_init or int(self.n/2), replace=False)]= True

        return tmp

    def ranges(self):
        return np.repeat([[0, 1]], self.dim(), axis=0)
    
    def cap(self, vector):
        return self.vectorize(self.devectorize(vector))

class GroupedBinaryVectorParameter(ParameterBase, ParameterSpace):
    def __init__(self, length, n_lower=None, n_upper=None, n_init=None, groups=None, random_state=None):
        """
        groups are expected to be represented by a dictionary of indices
        n_init, n_lower and n_upper are relative to the number of groups
        """
        self.length= length
        self.n_init= n_init
        self.n_lower= n_lower
        self.n_upper= n_upper
        self.groups= groups
        self.group_counts= {g: len(groups[g]) for g in groups}
        self.n= self.length
        self.n_groups= len(groups)
        ParameterBase.__init__(self)
        ParameterSpace.__init__(self, {'grouped_binary_vector': self}, random_state)
    
    def sample(self):
        tmp= np.repeat(False, self.length)
        group_mask= self.random_state.choice(list(self.groups.keys()), self.n_init or int(self.n_groups/2)+1)
        for g in group_mask:
            tmp[self.groups[g]]= True
        return tmp

    def dim(self):
        return self.n
    
    def vectorize(self, parameter):
        tmp= np.repeat(False, self.n_groups)
        i= 0
        for g in self.groups:
            tmp[i]= parameter[self.groups[g][0]] 
            i+= 1
        
        return tmp.astype(float)
    
    def devectorize(self, vector):
        tmp= np.repeat(False, self.length)

        binary_vector= vector >= 0.5

        if not self.n_lower is None or not self.n_upper is None:
            n_lower= self.n_lower or 0
            n_upper= self.n_upper or self.length

            if sum(binary_vector) >= n_lower and sum(binary_vector) <= n_upper:
                pass
            else:
                sort= sorted(vector)
                
                n= int((n_lower + n_upper)/2)
                binary_vector= vector >= sort[n]
                if sum(binary_vector) == 0:
                    binary_vector= np.repeat(False, len(binary_vector))
                    binary_vector[self.random_state.choice(list(range(len(binary_vector))), self.n_init or int(len(binary_vector)/2), replace=False)]= True

        if sum(binary_vector) == 0:
            binary_vector= np.repeat(False, len(binary_vector))
            binary_vector[self.random_state.choice(list(range(len(binary_vector))), self.n_init or int(len(binary_vector)/2), replace=False)]= True

        i= 0
        for g in self.groups:
            tmp[self.groups[g]]= binary_vector[i]
            i+= 1

        return tmp
    
    def mutate(self, parameter, rate= None):
        new_parameter= self.vectorize(parameter)
        current_n= np.sum(new_parameter)

        if rate == None:
            increase, decrease= False, False

            if (self.n_lower is None and current_n > 0) or (not self.n_lower is None and current_n > self.n_lower):
                decrease= True
            if (self.n_upper is None and current_n < self.n_groups - 1) or (not self.n_upper is None and current_n < self.n_upper):
                increase= True
            
            if decrease and increase:
                idx= self.random_state.choice(list(range(self.n_groups)))
            elif decrease:
                idx= self.random_state.choice(np.where(new_parameter == True)[0])
            elif increase:
                idx= self.random_state.choice(np.where(new_parameter == False)[0])
            
            new_parameter[idx]= not new_parameter[idx]
        else:
            for i in range(len(new_parameter)):
                if self.random_state.random() < rate:
                    new_parameter[i]= not new_parameter[i]
            if not self.n_lower is None and sum(new_parameter) < self.n_lower:
                new_parameter[self.random_state.choice(np.where(new_parameter == False)[0], self.n_lower - sum(new_parameter), replace=False)]= True
            elif not self.n_upper is None and sum(new_parameter) > self.n_upper:
                new_parameter[self.random_state.choice(np.where(new_parameter == True)[0], sum(new_parameter) - self.n_upper, replace=False)]= False

        binary_vector= new_parameter
        if sum(binary_vector) == 0:
            binary_vector= np.repeat(False, len(binary_vector))
            binary_vector[self.random_state.choice(list(range(len(binary_vector))), self.n_init or int(len(binary_vector)/2), replace=False)]= True

        return self.devectorize(new_parameter)
    
    def crossover(self, parameter1, parameter2, crossover_rate):
        new_parameter= self.vectorize(self.sample())
        parameter1= self.vectorize(parameter1)
        parameter2= self.vectorize(parameter2)

        for i in range(len(parameter2)):
            if crossover_rate is None or self.random_state.random() < crossover_rate:
                new_parameter[i]= parameter2[i] if self.random_state.randint(2) == 1 else parameter1[i]
        
        current_n= np.sum(new_parameter)

        increase, decrease= 0, 0
        if self.n_lower is None and current_n == 0:
            increase= 1
        if not self.n_lower is None and current_n < self.n_lower:
            increase= self.n_lower - current_n
        if not self.n_upper is None and current_n > self.n_upper:
            decrease= current_n - self.n_upper
        
        if increase > 0:
            already_set= set(np.where(new_parameter == True)[0])
            set_1= set(np.where(parameter1 == True)[0])
            set_2= set(np.where(parameter2 == True)[0])
            not_set_1= set_1.difference(already_set)
            not_set_2= set_2.difference(already_set)
            new_parameter[self.random_state.choice(list(not_set_1.union(not_set_2)), increase)]= True
        if decrease > 0:
            already_set= np.where(new_parameter == True)[0]
            new_parameter[self.random_state.choice(already_set, decrease)]= False

        binary_vector= new_parameter
        if sum(binary_vector) == 0:
            binary_vector= np.repeat(False, len(binary_vector))
            binary_vector[self.random_state.choice(list(range(len(binary_vector))), self.n_init or int(len(binary_vector)/2), replace=False)]= True

        return self.devectorize(new_parameter)
    
    def ranges(self):
        return np.repeat([[0, 1]], self.dim(), axis=0)
    
    def cap(self, vector):
        return self.vectorize(self.devectorize(vector))

class JointParameterSpace(ParameterSpace):
    def __init__(self, parameter_spaces):
        self.parameter_spaces= parameter_spaces
    
    def dim(self):
        return np.sum([self.parameter_spaces[p].dim() for p in self.parameter_spaces])

    def sample(self):
        result= {}
        for p in self.parameter_spaces:
            result[p]= self.parameter_spaces[p].sample()
        return result
    
    def sample_vectorized(self):
        return np.hstack([self.parameter_spaces[p].sample_vectorized() for p in self.parameter_spaces])

    def vectorize(self, parameters):
        return np.hstack([self.parameter_spaces[p].vectorize(parameters[p]) for p in parameters])
    
    def devectorize(self, vector):
        result= {}
        n= 0
        for p in self.parameter_spaces:
            result[p]= self.parameter_spaces[p].devectorize(vector[n:(n+self.parameter_spaces[p].dim())])
            n+= self.parameter_spaces[p].dim()
        
        return result
    
    def mutate(self, parameters, rate= 0.5):
        new_parameters= {}
        for p in self.parameter_spaces:
            new_parameters[p]= self.parameter_spaces[p].mutate(parameters[p])
        
        return new_parameters
    
    def crossover(self, parameters1, parameters2, crossover_rate= 0.5):
        new_parameters= {}
        for p in self.parameter_spaces:
            new_parameters[p]= self.parameter_spaces[p].crossover(parameters1[p], parameters2[p], 0.5)

        return new_parameters

    def ranges(self):
        return np.vstack([self.parameter_spaces[p].ranges() for p in self.parameter_spaces])

    def cap(self, vector):
        return self.vectorize(self.devectorize(vector))
    
    #@classmethod
    #def 