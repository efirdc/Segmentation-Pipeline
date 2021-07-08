import torch
import torchio as tio


class FindInterestingSlice(tio.Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kwargs = kwargs

    def apply_transform(self, subject):
        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )

        for image in images:
            if not isinstance(image, tio.LabelMap):
                continue

            # One hot
            if "one_hot" in image and image['one_hot']:
                mask = torch.argmax(image.data, dim=0) != 0
            else:
                mask = image.data[0] != 0

            planes = ('Saggital', 'Coronal', 'Axial')

            image['interesting_slice_ids'] = interesting_slice_ids = {}
            for plane, where in zip(planes, torch.where(mask)):
                slice_ids, counts = torch.unique(where, return_counts=True)
                interesting_slice_ids_ids = torch.argsort(counts, descending=True)
                interesting_slice_ids[plane] = slice_ids[interesting_slice_ids_ids]

        return subject

    def is_invertible(self):
        return True
