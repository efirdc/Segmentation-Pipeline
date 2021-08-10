import numpy as np
import torchio as tio
from torchio.transforms.spatial_transform import SpatialTransform


class CropToMask(SpatialTransform):
    def __init__(
            self,
            label_map_name: str,
            label_id: int = 1,
            label_channel: int = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.label_map_name = label_map_name
        self.label_id = label_id
        self.label_channel = label_channel
        self.kwargs = kwargs
        self.args_names = ('label_map_name', 'label_id', 'label_channel')

    def apply_transform(self, subject):

        if self.label_map_name not in subject:
            return subject

        label_map = subject[self.label_map_name]
        mask = label_map.data[self.label_channel] == self.label_id

        W, H, D = mask.shape
        W_where, H_where, D_where = np.where(mask)
        cropping = (
            W_where.min(), W - W_where.max(),
            H_where.min(), H - H_where.max(),
            D_where.min(), D - D_where.max()
        )

        crop_transform = tio.Crop(cropping=cropping, **self.kwargs)

        subject = crop_transform(subject)

        return subject

    def is_invertible(self):
        return False
