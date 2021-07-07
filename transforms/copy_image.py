import copy

import torchio as tio


class CopyProperty(tio.Transform):
    def __init__(self, old_name, new_name, **kwargs):
        super().__init__(**kwargs)
        self.old_name = old_name
        self.new_name = new_name
        self.args_names = ('old_name', 'new_name',)

    def apply_transform(self, subject):

        if self.old_name not in subject:
            return subject

        prop = subject[self.old_name]
        subject[self.new_name] = copy.deepcopy(prop)

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        return CopyProperty(self.new_name, self.old_name)
