class GlobalDict:
    def __init__(self):
        self.dict = dict()
    
    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, item):
        return item in self.dict
    
    def items(self):
        return self.dict.items()
    
    def add(self, key, val):
        """
        Add key value pair
        """
        self.__dict__[key] = val
        

class GlobalTime:
    def __init__(self):
        self.__value = 0
    
    def tick(self):
        self.__value += 1

    @property
    def value(self):
        return self.__value