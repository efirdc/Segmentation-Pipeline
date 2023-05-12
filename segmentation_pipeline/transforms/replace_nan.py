import torchio as tio

class ReplaceNan(tio.Transform):
    def __init__(
        self,
        replace_val: float = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.replace_val = replace_val
        self.args_names = ('replace_val',)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in subject.get_images(True, self.include, self.exclude):
            image['data'][image['data'].isnan()] = self.replace_val

        return subject

    def is_invertible(self):
        return False