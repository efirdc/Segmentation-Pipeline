from typing import Dict, Any
from numbers import Number
from inspect import signature

from .utils import is_sequence


def get_nested_config(elem):
    """
    Recursively goes through all elements of a nested structure of sequences and dictionaries
    All Config objects are replaced with their get_config()
    Any non-basic data types are replaced with their str() representation.
    """
    if isinstance(elem, Dict):
        return {k: get_nested_config(v) for k, v in elem.items()}
    if is_sequence(elem):
        return [get_nested_config(v) for v in elem]
    if isinstance(elem, Config):
        return get_nested_config(elem.get_config())
    if isinstance(elem, Number) or isinstance(elem, str) or isinstance(elem, bool):
        return elem
    else:
        return str(elem)


class Config:
    """
    Interface for a class that has configuration to be stored.

    The default implementation of get_config has the following requirement:
    In the class that inherits Config, every arg passed into __init__ must be
    stored as a class property with the same name as the corresponding parameter name
    Example:
        >>> class MyClass(Config):
        >>>     def __init__(self, foo, bar, baz=None):
        >>>         # Store all args as class properties with the same name as the parameter
        >>>         # This satisfies the requirement for the default get_config()
        >>>         self.foo = foo
        >>>         self.bar = bar
        >>>         self.baz = baz

    """

    def get_config(self) -> Dict[str, Any]:
        sig = signature(self.__init__)
        param_names = list(sig.parameters.keys())
        for param_name in param_names:
            if param_name not in self.__dict__:
                raise RuntimeError(f"All parameters for __init__ must be saved "
                                   f"as class properties with the same name in order "
                                   f"to use default get_config(). The parameter {param_name} "
                                   f"was not saved.")
        config = {param_name: self.__dict__[param_name] for param_name in param_names}
        return config

    def get_nested_config(self) -> Dict[str, Any]:
        return get_nested_config(self)

    def __repr__(self) -> str:
        config = self.get_config()
        config_str = ", ".join([f"{param_name}={arg}" for param_name, arg in config.items()])
        return f"{self.__class__.__name__}({config_str})"
