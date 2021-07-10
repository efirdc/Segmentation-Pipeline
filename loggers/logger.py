from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def setup(self, context):
        raise NotImplementedError()

    @abstractmethod
    def save_context(self, context, save_path, iteration):
        raise NotImplementedError()

    @abstractmethod
    def log(self, log_dict):
        raise NotImplementedError()
