import json


class Config(object):
    """
    Config contains simulation's configuration informations from json file.
    """
    def init(self, config_file=None):
        self.dict = []
        if config_file:
            with open(config_file, 'r') as cf:
                self.dict = config_file

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, item):
        return item in self.dict

    def items(self):
        return self.dict.items()

    def add(self, key, value):
        """
        Add key value pair
        """
        self.__dict__[key] = value

