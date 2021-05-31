import torchio as tio
import torch
from typing import Sequence


class SplitImage(tio.Transform):
    def __init__(
            self,
            image_name: str,
            new_image_names: Sequence[str],
            new_image_channels: Sequence[int],
            **kwargs
    ):
        super().__init__(**kwargs)

        assert len(new_image_names) == len(new_image_channels), "The number of image names and number of " \
                                                                "channels specified must be the same."
        self.image_name = image_name
        self.new_image_names = new_image_names
        self.new_image_channels = new_image_channels
        self.args_names = ('image_name', 'new_image_names', 'new_image_channels',)

    def apply_transform(self, subject):
        target_image = subject[self.image_name]
        image_class = target_image.__class__

        target_image_splits = target_image.data.split(self.new_image_channels)

        for image_name, image_data in zip(self.new_image_names, target_image_splits):
            subject[image_name] = image_class(tensor=image_data)

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        from .concatenate_images import ConcatenateImages
        return ConcatenateImages(
            image_names=self.new_image_names,
            image_channels=self.new_image_channels,
            new_image_name=self.image_name
        )
