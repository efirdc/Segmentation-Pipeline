import torchio as tio
import torch
from typing import Sequence


class ConcatenateImages(tio.Transform):
    def __init__(
            self,
            image_names: Sequence[str],
            image_channels: Sequence[int],
            new_image_name: str,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert len(image_names) == len(image_channels), "The number of image names and number of " \
                                                        "channels specified must be the same."

        self.image_names = image_names
        self.image_channels = image_channels
        self.new_image_name = new_image_name
        self.args_names = ('image_names', 'image_channels', 'new_image_name',)

    def apply_transform(self, subject):
        images = [subject[image_name] for image_name in self.image_names]
        new_image_data = torch.cat([image.data for image in images])

        if isinstance(new_image_data, torch.IntTensor):
            new_image = tio.LabelMap(tensor=new_image_data)
        else:
            new_image = tio.ScalarImage(tensor=new_image_data)

        subject[self.new_image_name] = new_image

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        from .split_image import SplitImage
        return SplitImage(
            image_name=self.new_image_name,
            new_image_names=self.image_names,
            new_image_channels=self.image_channels
        )
