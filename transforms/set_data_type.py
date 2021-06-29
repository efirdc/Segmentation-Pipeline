import numpy as np
import torch
import torchio as tio


class SetDataType(tio.Transform):
    def __init__(
            self,
            data_type: torch.dtype,
            intensity_only: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.data_type = data_type
        self.intensity_only = intensity_only
        self.args_names = ('data_type', 'intensity_only',)

    def apply_transform(self, subject):

        for image in subject.get_images(self.intensity_only, self.include, self.exclude):
            image.set_data(image.data.to(self.data_type))

        return subject

    def is_invertible(self):
        return False
