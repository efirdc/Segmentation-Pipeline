import torchio as tio


class RenameImage(tio.Transform):
    def __init__(self, image_name, new_image_name, **kwargs):
        super().__init__(**kwargs)
        self.image_name = image_name
        self.new_image_name = new_image_name
        self.args_names = ('image_name', 'new_image_name',)

    def apply_transform(self, subject):
        image = subject[self.image_name]
        del subject[self.image_name]
        subject[self.new_image_name] = image

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        return RenameImage(self.new_image_name, self.image_name)
