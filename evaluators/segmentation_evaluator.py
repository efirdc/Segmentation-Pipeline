from typing import Sequence

import torch
import torchio as tio
import pandas as pd

from transforms import *
from .evaluator import Evaluator
from utils import as_list


class SegmentationEvaluator(Evaluator):
    """ Performs a segmentation evaluation between predicted and target label maps.

    The predicted and target label maps must have a ``Dict[str, int]`` property ``'label_values'``,
    which maps a label's name to its value. These ``'label_values'`` must be identical for both
    label maps.

    Args:
        prediction_label_name: Key to a ``tio.LabelMap`` in each ``tio.Subject`` that is evaluated.
            Typically this will be the output of a segmentation model.
        target_label_name: Key to a ``tio.LabelMap`` in each ``tio.Subject`` that is evaluated.
            Typically this will be the ground truth manually traced label.
        labels_to_output: Optionally restrict which labels are used in the evaluation.
            If ``None`` then all labels will be used.
        stats_to_output: Optionally restrict which statistics are output from the evaluation.
            If ``None`` then all statistics will be output.
        summary_stats: If ``True`` then a pandas ``DataFrame`` containing summary statistics
            will be included in the output dictionary under the key ``'summary_stats'``.
            Each row

    """
    def __init__(
            self,
            prediction_label_name: str,
            target_label_name: str,
            labels_to_output: Sequence[str] = None,
            stats_to_output: Sequence[str] = None,
            output_summary_stats: bool = True,
            output_per_subject_label_stats: bool = True
    ):
        self.prediction_label_name = prediction_label_name
        self.target_label_name = target_label_name
        self.labels_to_output = as_list(labels_to_output)
        self.stats_to_output = as_list(stats_to_output)
        self.output_summary_stats = output_summary_stats
        self.output_per_subject_label_stats = output_per_subject_label_stats

    def evaluate(self, subjects):
        label_values = subjects[0][self.prediction_label_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject.name for subject in subjects]

        transform = tio.Compose([
            CustomSequentialLabels(),
            CustomOneHot(len(label_values))
        ])

        prediction_label = torch.stack([
            transform(subject[self.prediction_label_name]).data for subject in subjects
        ]).bool()
        target_label = torch.stack([
            transform(subject[self.target_label_name]).data for subject in subjects
        ]).bool()

        spatial_dims = (2, 3, 4)

        TP = (target_label & prediction_label).sum(dim=spatial_dims)
        FP = (~target_label & prediction_label).sum(dim=spatial_dims)
        TN = (~target_label & ~prediction_label).sum(dim=spatial_dims)
        FN = (target_label & ~prediction_label).sum(dim=spatial_dims)
        stats = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

        stats['dice'] = 2 * TP / (2 * TP + FP + FN)
        stats['jaccard'] = TP / (TP + FP + FN)

        stats['sensitivity'] = TP / (TP + FN)
        stats['specificity'] = TN / (TN + FP)
        stats['precision'] = TP / (TP + FP)
        stats['recall'] = TP / (TP + FN)

        out_dict = {}

        if self.output_per_subject_label_stats:
            out_dict['per_subject_label_stats'] = per_subject_stats = {}

            for i, label_name in enumerate(label_values.keys()):
                if self.labels_to_output and label_name not in self.labels_to_output:
                    continue
                df = pd.DataFrame()
                df['subject'] = subject_names
                for output in self.stats_to_output:
                    df[output] = stats[output][:, i]
                per_subject_stats[label_name] = df

        if self.output_summary_stats:
            summary_stats = {
                stat_name: {'mean': stat.float().mean(dim=0).item(), 'std': stat.float().std(dim=0).item()}
                for stat_name, stat in stats.items()
                if not self.stats_to_output or stat_name in self.stats_to_output
            }

            df = pd.DataFrame()
            df['label_name'] = label_names
            for output in self.stats_to_output:
                df[f'{output}.mean'] = summary_stats[output]['mean']
                df[f'{output}.std'] = summary_stats[output]['std']
            out_dict['summary_stats'] = df

        return out_dict


