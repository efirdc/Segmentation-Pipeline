from .logger import Logger


class NonLogger(Logger):
    """ Use if you don't want to log anything. """
    def __init__(self):
        pass

    def setup(self, context):
        pass

    def save_context(self, context, save_path, iteration):
        pass

    def log(self, log_dict):
        pass
