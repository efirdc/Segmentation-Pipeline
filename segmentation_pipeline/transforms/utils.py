from typing import Sequence, Type

import torchio as tio


# TODO: Contribute this as a method to tio.Compose.
# as it doesnt make sense to call unless tio.Compose is the root transform
def filter_transform(
        transform: tio.Compose,
        include_types: Sequence[Type[tio.Transform]] = None,
        exclude_types: Sequence[Type[tio.Transform]] = None,
):
    if isinstance(transform, tio.Compose):
        return tio.Compose([
            filter_transform(t, include_types=include_types, exclude_types=exclude_types)
            for t in transform
            if isinstance(t, tio.Compose) or (
                    (include_types is None or any(isinstance(t, typ) for typ in include_types))
                    and
                    (exclude_types is None or not any(isinstance(t, typ) for typ in exclude_types))
            )
        ])
    return transform
