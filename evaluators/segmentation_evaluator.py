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

    The two labels do not necessarily have to be the output of a segmentation model and a ground
    truth label. This evaluation could also be used to test the similarity in output between
    two segmentation models, or two ground truth labels from different experts.

    The supported output statistics are the following:
    ``('TP', 'FP', 'TN', 'FN', 'dice', 'jaccard', 'sensitivity', 'specificity', 'precision', 'recall')``

    Summary statistics for any of the above can also be computed and output from the evaluation
    The following are supported:
    ``('mean', 'median', 'mode', 'std', 'min', 'max')``

    Args:
        prediction_label_name: Key to a ``tio.LabelMap`` in each ``tio.Subject`` that is evaluated.
            Typically this should point to the output of a segmentation model.
        target_label_name: Key to a ``tio.LabelMap`` in each ``tio.Subject`` that is evaluated.
            Typically this should point to the ground truth manually traced label.
        output_summary_stats: If ``True`` then a dictionary containing summary statistics
            will be included in the output dictionary under the key ``'summary_stats'``.
            Has the structure ``{summary_stat_name: {stat_name: {label_name: value}}}``
        output_subject_stats: If ``True`` then a dictionary containing tables with per-subject stats
             will be included in the output dictionary under the key ``'subject_stats'``.
             Has the structure ``{label_name: table}`` where ``table`` is a ``pd.DataFrame`` and
             each row corresponds to a subject and each column corresponds to a statistic.
        labels_to_output: Optionally provide a sequence of label names to
            restrict which labels are used in the evaluation. If ``None`` then all labels will be used.
        stats_to_output: Optionally provide a sequence of statistic names to restrict which statistics
            are output from the evaluation.
            If ``None`` then all statistics will be output.
        summary_stats_to_output: Optionally provide a sequence of summary statistic names to restrict
            which summary stats are output from the evaluation.
            If ``None`` then all summary statistics will be output.
    """
    def __init__(
            self,
            prediction_label_name: str,
            target_label_name: str,
            output_summary_stats: bool = True,
            output_subject_stats: bool = True,
            labels_to_output: Sequence[str] = None,
            stats_to_output: Sequence[str] = None,
            summary_stats_to_output: Sequence[str] = None,
    ):
        self.prediction_label_name = prediction_label_name
        self.target_label_name = target_label_name
        self.output_summary_stats = output_summary_stats
        self.output_subject_stats = output_subject_stats
        self.labels_to_output = as_list(labels_to_output)
        self.stats_to_output = as_list(stats_to_output)
        self.summary_stats_to_output = as_list(summary_stats_to_output)

    def __call__(self, subjects):
        label_values = subjects[0][self.prediction_label_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject.name for subject in subjects]

        transform = tio.Compose([
            CustomSequentialLabels(),
            CustomOneHot(len(label_values))
        ])

        # Transform and stack prediction and target labels into one-hot (N, C, W, H, D) tensors
        prediction_label = torch.stack([
            transform(subject[self.prediction_label_name]).data for subject in subjects
        ]).bool()
        target_label = torch.stack([
            transform(subject[self.target_label_name]).data for subject in subjects
        ]).bool()

        # Compute (N, C) tensors for each statistic. Each element corresponds to one subject and one label.
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

        summary_stat_aggregators = {
            'mean': lambda x: torch.mean(x, dim=0),
            'median': lambda x: torch.median(x, dim=0).values,
            'mode': lambda x: torch.mode(x, dim=0).values,
            'std': lambda x: torch.std(x, dim=0),
            'min': lambda x: torch.min(x, dim=0).values,
            'max': lambda x: torch.max(x, dim=0).values,
        }

        labels_to_output = self.labels_to_output if self.labels_to_output else label_names
        stats_to_output = self.stats_to_output if self.stats_to_output else list(stats.keys())
        summary_stats_to_output = self.summary_stats_to_output if self.summary_stats_to_output \
            else list(summary_stat_aggregators.keys())

        if self.output_summary_stats:
            # Convert the stats dict to a summary stats dict,
            # i.e. apply the summary stat aggregators on the subject axis for each (N, C) stat tensor,
            # Recall that the desired form
            # {summary_stat_name: {stat_name: {label_name: value}}}
            out_dict['summary_stats'] = {
                summary_stat_name: {
                    stat_name: {
                        label_name: label_stat for label_name, label_stat
                        in zip(label_names, aggregator(stat.float()).tolist())
                        if label_name in labels_to_output
                    }
                    for stat_name, stat in stats.items()
                    if stat_name in stats_to_output
                }
                for summary_stat_name, aggregator in summary_stat_aggregators.items()
                if summary_stat_name in summary_stats_to_output
            }

        if self.output_subject_stats:

            # Desired key structure is {label_name: table}
            # where table is a pd.DataFrame with col=stat_name row_index=subject_name
            subject_stats = {}

            for i, label_name in enumerate(label_values.keys()):
                if label_name not in labels_to_output:
                    continue

                df = pd.DataFrame()
                df['subject'] = subject_names
                for output in stats_to_output:
                    df[output] = stats[output][:, i]
                subject_stats[label_name] = df

            out_dict['subject_stats'] = subject_stats

        return out_dict

    def class_name(self) -> str:
        return "segmentation_evaluator"
