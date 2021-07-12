import torchio as tio


class MinSizePad(tio.Transform):
    def __init__(self, min_size, **kwargs):
        super().__init__(**kwargs)

        if isinstance(min_size, int):
            self.min_size = (min_size, min_size, min_size)
        elif isinstance(min_size, tuple):
            self.min_size = min_size
        else:
            raise KeyError("min_size must be an int or tuple") 
        self.kwargs = kwargs
        self.args_names = ('min_size',)

    def apply_transform(self, subject):


        _, W, H, D = subject.get_first_image().shape
        W_min, H_min, D_min = self.min_size


        W_pad, H_pad, D_pad = (0, 0), (0, 0), (0, 0)
        if W < W_min:
            W_pad = self.calcPadding(W_min, W)

        if H < H_min:
            H_pad = self.calcPadding(H_min, H)

        if D < D_min:
            D_pad = self.calcPadding(D_min, D)

        self.padding = (*W_pad, *H_pad, *D_pad)

        if self.padding > (0, 0, 0, 0, 0, 0):
            pad_transform = tio.Pad(self.padding, **self.kwargs)
            subject = pad_transform(subject)

        return subject


    def calcPadding(self, goal, current):
        diff = (goal - current)
        half = diff // 2

        if diff %2 == 0:
            return (half, half)
        else:
            return (half, half + 1)

