import torch
import torch.nn.functional as F

import torchio as tio
from torchio.transforms.transform import TypeMaskingMethod
from torchio.transforms.preprocessing.label.label_transform import LabelTransform

from typing import Sequence, Tuple, Union, Dict

# These transformations behave almost the same as RemapLabels, RemoveLabels, and SequentialLabels from torchio
# However an additional feature is supported to keep track of label names in the label map
# The LabelMap can have a Dict[str, int] attribute 'label_values' which maps the name of the label to it's id
# in the LabelMap. These custom versions of the label transformations will also update the ids in this dictionary.
# It is recommended that you use these versions for this pipeline.

TypeLabelID = Union[int, str]
TypeLabelRemapping = Union[Dict[int, int], Sequence[Tuple[str, int, int]]]


class CustomRemapLabels(LabelTransform):
    def __init__(
            self,
            remapping: TypeLabelRemapping,
            masking_method: TypeMaskingMethod = None,
            invertible: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.kwargs = kwargs
        self.remapping = self._parse_label_remapping(remapping)
        self.masking_method = masking_method
        self.invertible = invertible
        self.args_names = ('remapping', 'masking_method', 'invertible')

    def apply_transform(self, subject):
        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )

        for image in images:
            if not isinstance(image, tio.LabelMap):
                continue

            if isinstance(self.remapping, Dict):
                label_remapping = self.remapping
            elif isinstance(self.remapping, Sequence):
                label_remapping = {old_id: new_id for _, old_id, new_id in self.remapping}
                if "label_values" in image:
                    label_values = image['label_values']
                    for label_name, _, new_id in self.remapping:
                        label_values[label_name] = new_id
            else:
                raise RuntimeError(self._label_remapping_error(self.remapping))

            new_data = image.data.clone()
            mask = self.get_mask_from_masking_method(
                self.masking_method,
                subject,
                new_data,
            )
            for old_id, new_id in label_remapping.items():
                new_data[mask & (image.data == old_id)] = new_id
            image.set_data(new_data)

        return subject

    def is_invertible(self):
        return self.invertible

    def inverse(self):
        if isinstance(self.remapping, Dict):
            inverse_remapping = {v: k for k, v in self.remapping.items()}
        elif isinstance(self.remapping, Sequence):
            inverse_remapping = [(label_name, old_id, new_id) for label_name, new_id, old_id in self.remapping]
        else:
            raise RuntimeError(self._label_remapping_error(self.remapping))

        inverse_transform = CustomRemapLabels(
            inverse_remapping,
            masking_method=self.masking_method,
            **self.kwargs,
        )
        return inverse_transform

    def _parse_label_remapping(self, remapping: TypeLabelRemapping):
        if isinstance(remapping, Dict):
            for k, v, in remapping.items():
                if not isinstance(k, int) or not isinstance(v, int):
                    raise ValueError(self._label_remapping_error(remapping))
        elif isinstance(remapping, Sequence):
            for remap in remapping:
                if any(not isinstance(elem, t) for elem, t in zip(remap, (str, int, int))):
                    raise ValueError(self._label_remapping_error(remapping))
        else:
            raise ValueError(self._label_remapping_error(remapping))
        return remapping

    def _label_remapping_error(self, remapping: TypeLabelRemapping):
        return "Label remapping must be a Dict[int, int] that remaps old ids to new ids " \
               "or a Sequence[Tuple[str, int, int]] where each tuple is a (label_name, old_id, new_id), " \
               f"not {remapping} of type {type(remapping)}"


class CustomRemoveLabels(LabelTransform):
    def __init__(
            self,
            labels,
            background_label=0,
            masking_method=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.labels = labels
        self.background_label = background_label
        self.masking_method = masking_method
        self.args_names = ('labels', 'background_label', 'masking_method',)

    def apply_transform(self, subject):

        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for name, image in images_dict.items():
            if not isinstance(image, tio.LabelMap):
                continue

            labels = []
            for label in self.labels:
                if isinstance(label, int):
                    labels.append(label)
                elif isinstance(label, str):
                    if 'label_values' not in image:
                        raise RuntimeError(f'Image must have a Dict[str, int] property '
                                           f'with the key "label_values" in order to '
                                           f'remove a label by its name.')
                    labels.append(image['label_values'][label])
                else:
                    raise ValueError(f'Label to remove must be a string or an int, '
                                     f'not {label} of type {type(label)}.')

            remapping = {label: self.background_label for label in labels}
            transform = CustomRemapLabels(
                remapping=remapping,
                masking_method=self.masking_method,
                include=name,
                invertible=False
            )
            subject = transform(subject)

            if "label_values" not in image:
                continue

            for label_name, label_value in image['label_values'].items():
                if label_value in labels:
                    del image[label_name]

        return subject

    def is_invertible(self):
        return False


class CustomSequentialLabels(LabelTransform):
    def __init__(self, masking_method=None, **kwargs):
        super().__init__(**kwargs)
        self.masking_method = masking_method
        self.args_names = ('masking_method',)

    def apply_transform(self, subject):
        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for name, image in images_dict.items():
            if not isinstance(image, tio.LabelMap):
                continue

            if 'label_values' in image:
                label_values = image['label_values']
                label_names = list(label_values.keys())
                label_names.sort(key=lambda label_name: label_values[label_name])
                remapping = [
                    (label_name, label_values[label_name], i + 1)
                    for i, label_name in enumerate(label_names)
                ]
            else:
                unique_labels = torch.unique(image.data, sorted=True).tolist()
                unique_labels = unique_labels[1:]  # Remove the 0
                remapping = {
                    unique_labels[i]: i + 1
                    for i in range(len(unique_labels))
                }

            transform = CustomRemapLabels(
                remapping=remapping,
                masking_method=self.masking_method,
                include=name,
            )
            subject = transform(subject)

        return subject


class CustomOneHot(LabelTransform):
    def __init__(
        self,
        num_classes: int = -1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.args_names = ('num_classes',)

    def apply_transform(self, subject):

        for image in self.get_images(subject):
            num_channels = image.data.shape[0]
            if num_channels != 1:
                message = (
                    f'The number of input channels was expected to be {1},'
                    f' but it is {num_channels}'
                )
                raise RuntimeError(message)

            if self.num_classes == -1 and 'label_values' in image:
                num_classes = max(image['label_values'].values()) + 1
            else:
                num_classes = self.num_classes

            data = image.data[0]
            one_hot = F.one_hot(data.long(), num_classes=num_classes)
            image.set_data(one_hot.permute(3, 0, 1, 2).type(data.type()))

            image['one_hot'] = True

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        return CustomArgMax(num_classes=self.num_classes, **self.kwargs)


class CustomArgMax(LabelTransform):
    def __init__(
        self,
        num_classes: int = -1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.args_names = ('num_classes',)

    def apply_transform(self, subject):

        for image in self.get_images(subject):
            new_data = torch.argmax(image.data, dim=0, keepdim=True)
            image.set_data(new_data)

            image['one_hot'] = False

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        return CustomOneHot(num_classes=self.num_classes, **self.kwargs)


class MergeLabels(LabelTransform):
    def __init__(
        self,
        merge_labels: Sequence[Tuple[str, str]],
        left_masking_method=None,
        right_masking_method=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if (left_masking_method is None and right_masking_method is None) or \
           (left_masking_method is not None and right_masking_method is not None):
            raise ValueError('One of left_masking_method or right_masking_method must be provided, '
                             'but not both.')
        for left_label, right_label in merge_labels:
            if not isinstance(left_label, str) or not isinstance(right_label, str):
                raise ValueError("Label identifiers must be strings.")
        self.merge_labels = merge_labels
        self.left_masking_method = left_masking_method
        self.right_masking_method = right_masking_method
        self.args_names = ('merge_labels', 'left_masking_method', 'right_masking_method')

    def apply_transform(self, subject):
        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for name, image in images_dict.items():
            if not isinstance(image, tio.LabelMap):
                continue

            merge_labels = self.merge_labels
            if 'label_values' not in image:
                raise RuntimeError(f"label_values dict not found in image {image} for subject {subject}")
            label_values = image['label_values']

            if self.left_masking_method:
                remapping = [
                    (left_label, label_values[left_label], label_values[right_label])
                    for left_label, right_label in merge_labels
                ]
                masking_method = self.left_masking_method
            elif self.right_masking_method:
                remapping = [
                    (right_label, label_values[right_label], label_values[left_label])
                    for left_label, right_label in merge_labels
                ]
                masking_method = self.right_masking_method
            else:
                raise RuntimeError()

            transform = CustomRemapLabels(
                remapping=remapping,
                masking_method=masking_method,
                include=name,
            )
            subject = transform(subject)

        return subject

    def is_invertible(self):
        return False
