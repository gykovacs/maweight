import os
import os.path
import json
import jsonlines

__all__= ['CacheBase']

class CacheBase():
    def __init__(self, cache_path=None):
        self.cache= {}
        self.cache_path= cache_path
        if self.cache_path:
            if not os.path.exists(self.cache_path):
                if not os.path.exists(os.path.dirname(self.cache_path)):
                    os.mkdir(self.cache_path)
            else:
                with jsonlines.open(self.cache_path) as reader:
                    for obj in reader:
                        self.cache[obj['parameters']]= obj['value']
            self.cache_writer= jsonlines.open(self.cache_path, mode='a')
    
    def is_in_cache(self, parameter):
        return parameter in self.cache

    def get_from_cache(self, parameter):
        if self.is_in_cache(parameter):
            return self.cache[parameter]
        else:
            return None
    
    def put_into_cache(self, parameter, value):
        self.cache[parameter]= value
        if self.cache_path:
            self.cache_writer.write({'parameters': parameter, 'value': value})
    
    def close_cache(self):
        if self.cache_path:
            self.cache_writer.close()