from typing import Union, Optional
from statistics import mean, median

import torchio as tio
from torchio.transforms.preprocessing.spatial.resample import TypeSpacing


class TargetResample(tio.Resample):
    def __init__(
            self,
            target_spacing: Union[TypeSpacing, str],
            tolerance: TypeSpacing,
            image_interpolation: str = 'linear',
            pre_affine_name: Optional[str] = None,
            scalars_only: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.spacing_modes = {
            'mean': mean, 'median': median, 'min': min, 'max': max
        }
        if isinstance(target_spacing, str) and target_spacing not in self.spacing_modes.keys():
            raise ValueError(f"Spacing mode must be one of: {tuple(self.spacing_modes.keys())}")
        else:
            target_spacing = tio.Resample.parse_spacing(target_spacing)

        self.target_spacing = target_spacing
        self.tolerance = tio.Resample.parse_spacing(tolerance)
        self.image_interpolation = image_interpolation
        self.pre_affine_name = pre_affine_name
        self.scalars_only = scalars_only

        self.kwargs = kwargs
        self.args_names = ('target_spacing', 'tolerance')

    def apply_transform(self, subject):
        current_spacing = subject.get_first_image().spacing

        if isinstance(self.target_spacing, str):
            target_spacing = self.spacing_modes[self.target_spacing](current_spacing)
            target_spacing = (target_spacing, target_spacing, target_spacing)
        else:
            target_spacing = self.target_spacing

        # No operation if all current spacings are within tolerance of the target spacing
        if all(abs(cur - tar) < tol for cur, tar, tol in zip(current_spacing, target_spacing, self.tolerance)):
            return subject

        # Iteratively scale
        new_spacing = []
        for cur, tar, tol in zip(current_spacing, target_spacing, self.tolerance):
            step = 1

            spacing = cur

            while abs(spacing - tar) > tol:

                if cur < tar:
                    scale = tar / cur
                    scale = round(scale * step) / step
                else:
                    scale = cur / tar
                    scale = 1 / (round(scale * step) / step)

                spacing = cur * scale
                step += 1

            new_spacing.append(spacing)

        new_spacing = tuple(new_spacing)
        resample = tio.Resample(target=new_spacing,
                                image_interpolation=self.image_interpolation,
                                pre_affine_name=self.pre_affine_name,
                                scalars_only=self.scalars_only,
                                **self.kwargs)
        subject = resample(subject)

        return subject


