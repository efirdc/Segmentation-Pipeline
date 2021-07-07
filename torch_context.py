import inspect
import os
from datetime import datetime
from pprint import pformat
from typing import Dict, Any

import torch
import dill


class TorchContext:
    """ A simple entity-component system for pytorch experiments.

    Args:
        device: A pytorch device. All components of type torch.nn.Module are automatically transferred to
            the device when the component is initialized.
        name: A name for a new context
        file_path: A file path to a previously saved context.
            Note: Either name or file_path must be provided, but not both.
        variables: A dictionary that is added to the system environment variables.
            Environment variables in component arguments will be expanded by the TorchContext.
        metadata: A dictionary of additional metadata that is saved with the context.

    Example:
        >>> # Anything that is environment dependent (like dataset paths) should be set as an environment variable
        >>> # This ensures dataset paths are not hard-coded, and the TorchContext can be shared.
        >>> variables = {
        ...     "DATASET_PATH": "C:/Datasets/CatsAndDogs/"
        ...     "CHECKPOINTS_PATH": "C:/Checkpoints/"
        ... }
        >>>
        >>> # Initialize the TorchContext
        >>> context = TorchContext(torch.device('cuda'), name='experiment-00007562', variables=variables)
        >>>
        >>> # Set a components name, constructor, followed by its params as key word arguments.
        >>> # Environment variables are expanded when the component is initialized
        >>> context.add_component('dataset', LabeledImageDataset, root="$DATASET_PATH")
        >>> context.add_component('model', Classifier, num_classes=2, base_features=256)
        >>>
        >>> # Any argument that starts with 'self.' is passed into eval() inside the context.
        >>> # This can be used so that the optimizer can get the model parameters
        >>> context.add_component('optimizer', Adam, params="self.model.parameters()", lr=0.01)
        >>> context.add_component('criterion', CrossEntropyLoss)
        >>> context.add_component('trainer', ClassifierTrainer, checkpoints_path="$CHECKPOINTS_PATH")
        >>>
        >>> # Initialize all components, nn.Modules are transferred to the context device.
        >>> context.init_components()
        >>>
        >>> # Initialized components are now accessible as context.component_name
        >>> context.trainer.train(context, iterations=1000)
    """
    def __init__(
            self,
            device: torch.device,
            name: str = None,
            file_path: str = None,
            variables: Dict[str, str] = None,
            metadata: Dict[str, Any] = None
    ):
        assert (name is None) != (file_path is None), "Either provide a name to create a new context, " \
                                                      "or a file_path to load an existing context, but not both."
        self.device = device
        self.name = name
        self.variables = {} if variables is None else variables
        self.metadata = {} if metadata is None else metadata

        self.creation_time = datetime.now().strftime("%y%m%d-%H%M%S")
        self.component_definitions = []
        self.file_paths = []

        if file_path is not None:
            checkpoint = torch.load(file_path)

            self.name = checkpoint["name"]
            self.component_definitions = checkpoint['component_definitions']
            self.creation_time = checkpoint["creation_time"]

            for var, value in checkpoint["variables"].items():
                if var not in self.variables and var not in os.environ:
                    raise Warning(f"An environment variable ${var} was defined as an input to this context. "
                                  f"However it was not found in the variables parameter or the system environment "
                                  f"variables. The previously used value {value} will be used instead, but a "
                                  f"component may not initialize properly.")
            checkpoint['variables'].update(self.variables)
            self.variables = checkpoint["variables"]

            self.file_paths = checkpoint["file_paths"]
            self.metadata = checkpoint["metadata"]

        if self.variables is not None:
            os.environ.update(variables)

        self.loaded = False

    def init_components(self):
        self._enforce_not_loaded()

        for component_initializer in self.component_definitions:
            self._init_component(component_initializer)

        self.loaded = True

    def _init_component(self, component_definition):
        name = component_definition["name"]
        constructor = component_definition["constructor"]
        params = self._fix_params(component_definition["params"])

        component = constructor(**params)

        if "state_dict" in component_definition:
            component.load_state_dict(component_definition["state_dict"])

        if isinstance(component, torch.nn.Module):
            component = component.to(self.device)

        self.__dict__[name] = component

    def add_component(self, name, constructor, **params):
        self._enforce_not_loaded()

        component_definition = dict(name=name, constructor=constructor, params=params)
        self.component_definitions.append(component_definition)

        try:
            component_file_path = inspect.getsourcefile(constructor)
            self.file_paths.append(component_file_path)
        except TypeError:
            pass

        if self.loaded:
            self._init_component(component_definition)

    def update_component(self, name, constructor=None, **params):
        self._enforce_not_loaded()

        for i, defn in enumerate(self.component_definitions):
            if defn['name'] == name:
                if constructor is not None:
                    defn['constructor'] = constructor
                defn['params'].update(params)
                return

        raise ValueError(f"Could not find component {name} in the context.")

    def keep_components(self, names):
        self._enforce_not_loaded()

        self.component_definitions = [
            defn for defn in self.component_definitions
            if defn['name'] in names
        ]

    def remove_components(self, names):
        self._enforce_not_loaded()

        self.component_definitions = [
            defn for defn in self.component_definitions
            if defn['name'] in names
        ]

    def remove_component(self, name):
        return self.remove_components([name])

    def _fix_params(self, params):
        if isinstance(params, dict):
            return {k: self._fix_params(v) for k, v in params.items()}
        if isinstance(params, list):
            return [self._fix_params(param) for param in params]
        if isinstance(params, tuple):
            return tuple(self._fix_params(param) for param in params)

        param = params
        if isinstance(param, str):

            if param.startswith("self."):
                return eval(param)

            param = os.path.expandvars(param)
            if '$' in param:
                raise Warning(f"Environment variable found in argument {param} was not expanded. "
                              f"A component may not initialize correctly. Either set the variable "
                              f"in your OS or pass it into the context in the variables dict.")

        return param

    def save(self, filename):

        # If a component has a .state_dict() function, then save it in the component definition.
        for component_definition in self.component_definitions:
            component = self.__dict__[component_definition["name"]]
            if hasattr(component, "state_dict"):
                component_definition["state_dict"] = component.state_dict()

        checkpoint = dict(
            name=self.name,
            component_definitions=self.component_definitions,
            creation_time=self.creation_time,
            variables=self.variables,
            file_paths=self.file_paths,
            metadata=self.metadata
        )

        torch.save(checkpoint, filename, pickle_module=dill)

    def _enforce_not_loaded(self):
        if self.loaded:
            raise NotImplementedError("Modifying components after they are initialized is not supported.")

    def get_config(self, component_names):
        out = {}

        for defn in self.component_definitions:
            if defn['name'] not in component_names:
                continue
            out.update({
                f"{defn['name']}.{param}": value
                for param, value in defn["params"].items()
                if any(isinstance(value, t) for t in (str, int, float))
            })

        return out

    def __repr__(self):
        out = f'TorchContext {self.name} created at {self.creation_time}\n'
        for i, component_definition in enumerate(self.component_definitions):
            out += f'\ncomponent_id={i}\n'
            out += f'component_defintition={pformat(component_definition, 4)}\n'
            component = self.__dict__[component_definition['name']] if self.loaded else 'not loaded'
            out += f'component={component}\n'
        return out
