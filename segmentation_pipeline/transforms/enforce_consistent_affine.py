import torchio as tio


class EnforceConsistentAffine(tio.Transform):
    def __init__(
            self,
            source_image_name: str = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.source_image_name = source_image_name
        self.args_names = ('source_image_name',)

    def apply_transform(self, subject):
        if self.source_image_name is not None and self.source_image_name not in subject:
            return subject

        if self.source_image_name is not None:
            source_image = subject[self.source_image_name]
        else:
            source_image = subject.get_first_image()
        images_dict = subject.get_images_dict(intensity_only=False, include=self.include, exclude=self.exclude)

        for image_name, image in images_dict.items():
            if self.source_image_name == image_name:
                continue
            image.affine = source_image.affine

        return subject

    def is_invertible(self):
        return False
