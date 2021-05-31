import torchio as tio
from torchio.transforms.preprocessing.label.label_transform import LabelTransform
import torch

# These transformations behave almost the same as RemapLabels, RemoveLabels, and SequentialLabels from torchio
# However an additional feature is supported to keep track of label names in the label map
# The LabelMap can have a Dict[str, int] attribute 'label_names' which maps the name of the label to it's id
# in the LabelMap. These custom versions of the label transformations will also update the ids in this dictionary.
# It is recommended that you use these versions for this pipeline.


class CustomRemapLabels(LabelTransform):
    def __init__(self, remapping, masking_method=None, inversed=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.remapping = remapping
        self.masking_method = masking_method
        self.inversed = inversed
        self.args_names = ('remapping', 'masking_method', 'inversed',)

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
            # Also filter out any removed labels
            if self.inversed:
                old_label_names = image["label_names_hist"].pop()
                image['label_names'] = {name: label for name, label in old_label_names.items() if name in label_names}
                continue

            # On the forward pass, apply the remapping to the label_names
            image["label_names_hist"].append(label_names.copy())
            for name, label in label_names.items():
                if label in self.remapping:
                    label_names[name] = self.remapping[label]

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        inverse_remapping = {v: k for k, v in self.remapping.items()}
        inverse_transform = CustomRemapLabels(
            inverse_remapping,
            masking_method=self.masking_method,
            inversed=(not self.inversed),
            **self.kwargs,
        )
        return inverse_transform


class CustomRemoveLabels(tio.RemoveLabels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_transform(self, subject):
        super().apply_transform(subject)

        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )

        for image in images:
            if not isinstance(image, tio.LabelMap):
                continue

            if "label_names" not in image:
                continue

            image['label_names'] = {
                name: label for name, label in image['label_names'].items()
                if label not in self.labels
            }

        return subject

    def is_invertible(self):
        return False


class CustomSequentialLabels(LabelTransform):
    def __init__(self, masking_method=None, **kwargs):
        super().__init__(**kwargs)
        self.masking_method = masking_method

    def apply_transform(self, subject):
        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for name, image in images_dict.items():
            if not isinstance(image, tio.LabelMap):
                continue

            unique_labels = torch.unique(image.data)
            remapping = {
                unique_labels[i].item(): i
                for i in range(1, len(unique_labels))
            }
            transform = CustomRemapLabels(
                remapping=remapping,
                masking_method=self.masking_method,
                include=name,
            )
            subject = transform(subject)

        return subject
