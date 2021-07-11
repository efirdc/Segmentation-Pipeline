from statistics import mean, median

import torchio as tio


class IsotropicResample(tio.SpatialTransform):
    def __init__(
            self,
            spacing_mode: str,
            tolerance: float,
            min_spacing: float = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.spacing_modes = {
            'mean': mean, 'median': median, 'min': min, 'max': max
        }
        if spacing_mode not in self.spacing_modes.keys():
            raise ValueError(f"Spacing mode must be one of: {tuple(spacing_mode.keys())}")
        if tolerance < 0:
            raise ValueError(f"Tolerance value must be a positive number, not {tolerance}")
        self.spacing_mode = spacing_mode
        self.tolerance = tolerance
        self.kwargs = kwargs
        self.min_spacing = min_spacing
        self.args_names = ('spacing_mode', 'tolerance', 'min_spacing')

    def apply_transform(self, subject):
        spacing = subject.get_first_image().spacing
        isotropic_spacing = self.spacing_modes[self.spacing_mode](spacing)

        if self.min_spacing:
            increment = isotropic_spacing * 0.5
            while isotropic_spacing < self.min_spacing:
                isotropic_spacing += increment

        new_spacing = []
        for s in spacing:
            if abs(s - isotropic_spacing) <= self.tolerance:
                new_spacing.append(s)

            # Try to get within tolerance using integer multiples/divisors of the current spacing
            elif s < isotropic_spacing:
                s = s * round(isotropic_spacing / s)
                if abs(s - isotropic_spacing) <= self.tolerance:
                    new_spacing.append(s)
                else:
                    new_spacing.append(isotropic_spacing)
            elif s > isotropic_spacing:
                s = s / round(s / isotropic_spacing)
                if abs(s - isotropic_spacing) <= self.tolerance:
                    new_spacing.append(s)
                else:
                    new_spacing.append(isotropic_spacing)
        new_spacing = tuple(new_spacing)

        # Don't resample if voxels are already isotropic within tolerance parameter
        if spacing == new_spacing:
            return subject

        resample = tio.Resample(new_spacing, **self.kwargs)
        subject = resample(subject)
        return subject


