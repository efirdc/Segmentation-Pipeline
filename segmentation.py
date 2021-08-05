from typing import Sequence, Dict, Union, Any, Optional

import torchio as tio
from torchio.transforms.preprocessing.label.label_transform import LabelTransform

from utils import filter_transform
from transforms import *


def add_evaluation_labels(subjects: Sequence[tio.Subject]):
    for subject in subjects:
        transform = subject.get_composed_history()
        label_transform_types = [LabelTransform, CopyProperty, RenameProperty, ConcatenateImages]
        label_transform = filter_transform(transform, include_types=label_transform_types)
        inverse_label_transform = label_transform.inverse(warn=False)

        evaluation_transform = tio.Compose([
            inverse_label_transform,
            CustomSequentialLabels(),
            filter_transform(inverse_label_transform, exclude_types=[CustomRemapLabels]).inverse(warn=False)
        ])

        if 'y_pred' in subject:
            pred_subject = tio.Subject({'y': subject['y_pred']})
            subject['y_pred_eval'] = evaluation_transform(pred_subject)['y']

        if 'y' in subject:
            target_subject = tio.Subject({'y': subject['y']})
            subject['y_eval'] = evaluation_transform(target_subject)['y']