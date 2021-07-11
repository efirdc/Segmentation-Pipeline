from typing import Sequence, Tuple, Union

import torchio as tio
from torchio.typing import TypeNumber
import torch


TypeLabelWeights = Tuple[str, Union[int, str], TypeNumber]


class ImageFromLabels(tio.Transform):
    def __init__(
            self,
            new_image_name: str,
            label_weights: Sequence[TypeLabelWeights],
            mode: str = 'overwrite',
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.new_image_name = new_image_name
        self.label_weights = label_weights
        self.mode = mode
        self.args_names = ('new_image_name', 'label_weights', 'mode')

    def apply_transform(self, subject):
        subject.check_consistent_spatial_shape()

        output_data = torch.zeros(1, *subject.spatial_shape)

        for label_map_name, label_identifier, weight in self.label_weights:

            if label_map_name not in subject:
                continue
            label_map = subject[label_map_name]

            if isinstance(label_identifier, str):
                if 'label_values' not in label_map:
                    raise RuntimeError(f'LabelMap must have a Dict[str, int] property '
                                       f'with the key "label_values" in order to '
                                       f'select a label by its name.')
                label_identifier = label_map['label_values'][label_identifier]

            label_data = label_map['data']
            if 'one_hot' in label_map and label_map['one_hot']:
                label_data = torch.argmax(label_map['data'], dim=0, keepdim=True)
            label_mask = label_data[0:1] == label_identifier

            if self.mode == "additive":
                output_data += label_mask.float() * weight
            if self.mode == "overwrite":
                output_data[label_mask] = weight

        affine = subject.get_first_image().affine
        subject[self.new_image_name] = tio.ScalarImage(tensor=output_data, affine=affine)

        return subject
