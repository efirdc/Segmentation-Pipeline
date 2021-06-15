import torch
import torch.nn.functional as F

import torchio as tio
from torchio.transforms.preprocessing.label.label_transform import LabelTransform

from typing import Sequence, Tuple, Union

# These transformations behave almost the same as RemapLabels, RemoveLabels, and SequentialLabels from torchio
# However an additional feature is supported to keep track of label names in the label map
# The LabelMap can have a Dict[str, int] attribute 'label_names' which maps the name of the label to it's id
# in the LabelMap. These custom versions of the label transformations will also update the ids in this dictionary.
# It is recommended that you use these versions for this pipeline.


class CustomRemapLabels(LabelTransform):
    def __init__(self, remapping, masking_method=None, inversed=False, invertible=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.remapping = remapping
        self.masking_method = masking_method
        self.inversed = inversed
        self.invertible = invertible
        self.args_names = ('remapping', 'masking_method', 'inversed', 'invertible')

    def apply_transform(self, subject):
        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )

        for image in images:
            if not isinstance(image, tio.LabelMap):
                continue

            new_data = image.data.clone()
            mask = self.get_mask_from_masking_method(
                self.masking_method,
                subject,
                new_data,
            )
            for old_id, new_id in self.remapping.items():
                new_data[mask & (image.data == old_id)] = new_id
            image.set_data(new_data)

            if "label_names" not in image:
                continue

            # Keep track of the label name history with a stack
            if "label_names_hist" not in image:
                image["label_names_hist"] = []

            label_names = image['label_names']

            # If this is the inverse pass, then pop label names off the history stack.
            # Also filter out any removed labels TODO: Is this filter necessary?
            if self.inversed:
                old_label_names = image["label_names_hist"].pop()
                image['label_names'] = {
                    name: label for name, label in old_label_names.items()
                    if name in label_names
                }
                continue

            # On the forward pass, apply the remapping to the label_names
            if self.invertible:
                image["label_names_hist"].append(label_names.copy())
            for name, label in label_names.items():
                if label in self.remapping:
                    label_names[name] = self.remapping[label]

        return subject

    def is_invertible(self):
        return self.invertible

    def inverse(self):
        inverse_remapping = {v: k for k, v in self.remapping.items()}
        inverse_transform = CustomRemapLabels(
            inverse_remapping,
            masking_method=self.masking_method,
            inversed=(not self.inversed),
            **self.kwargs,
        )
        return inverse_transform


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
                    if 'label_names' not in image:
                        raise RuntimeError(f'Image must have a Dict[str, int] "label_names" '
                                           'in order to remove a label by its name.')
                    labels.append(image['label_names'][label])
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

            if "label_names" not in image:
                continue

            image['label_names'] = {
                name: label for name, label in image['label_names'].items()
                if label not in labels
            }

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

            if 'label_names' in image:
                unique_labels = list(image['label_names'].values())
                unique_labels.sort()
            else:
                unique_labels = torch.unique(image.data).tolist()
            remapping = {
                unique_labels[i]: i
                for i in range(1, len(unique_labels))
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
        num_classes: [int, Sequence[int]],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = [num_classes] if isinstance(num_classes, int) else num_classes
        self.args_names = ('num_classes',)

    def apply_transform(self, subject):

        for image in self.get_images(subject):
            num_channels = image.data.shape[0]
            if num_channels != len(self.num_classes):
                message = (
                    f'The number of input channels was expected to be {len(self.num_classes)},'
                    f' but it is {num_channels}'
                )
                raise RuntimeError(message)

            new_data = []
            for i, num_classes in enumerate(self.num_classes):
                data = image.data[i]
                one_hot = F.one_hot(data.long(), num_classes=num_classes)
                new_data.append(one_hot.permute(3, 0, 1, 2).type(data.type()))
            image.set_data(torch.cat(new_data))

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        return CustomArgMax(num_classes=self.num_classes)


class CustomArgMax(LabelTransform):
    def __init__(
        self,
        num_classes: [int, Sequence[int]],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = [num_classes] if isinstance(num_classes, int) else num_classes
        self.args_names = ('num_classes',)

    def apply_transform(self, subject):

        for image in self.get_images(subject):
            if image.num_channels != sum(self.num_classes):
                message = (
                    f'The number of input channels was expected to be {sum(self.num_classes)},'
                    f' but it is {image.num_channels}'
                )
                raise RuntimeError(message)

            new_data = torch.split(image.data, self.num_classes)
            new_data = [torch.argmax(x, dim=0, keepdim=True) for x in new_data]
            new_data = torch.cat(new_data)
            image.set_data(new_data)

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        return CustomOneHot(num_classes=self.num_classes)


TypeLabelID = Union[int, str]


class MergeLabels(LabelTransform):
    def __init__(
        self,
        merge_labels: Sequence[Tuple[TypeLabelID, TypeLabelID]],
        left_masking_method=None,
        right_masking_method=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if (left_masking_method is None and right_masking_method is None) or \
           (left_masking_method is not None and right_masking_method is not None):
            raise ValueError('One of left_masking_method or right_masking_method must be provided, '
                             'but not both.')
        self.merge_labels = merge_labels
        self.left_masking_method = left_masking_method
        self.right_masking_method = right_masking_method

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
            if 'label_names' in image:
                label_names = image['label_names']

                def convert_label(label):
                    if isinstance(label, str):
                        return label_names[label]
                    if isinstance(label, int):
                        return label
                    else:
                        raise ValueError(f'Label identifiers must be string or an int, '
                                         f'not {label} of type {type(label)}.')
                merge_labels = [
                    (convert_label(left), convert_label(right))
                    for left, right in merge_labels
                ]

            if self.left_masking_method:
                remapping = {left: right for left, right in merge_labels}
                masking_method = self.left_masking_method
            else:
                remapping = {right: left for left, right in merge_labels}
                masking_method = self.right_masking_method

            transform = CustomRemapLabels(
                remapping=remapping,
                masking_method=masking_method,
                include=name,
            )
            subject = transform(subject)

        return subject
