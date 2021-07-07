from typing import Sequence

import torch
import torchio as tio
import pandas as pd

from transforms import *
from .evaluator import Evaluator


class SegmentationEvaluator(Evaluator):
    """ Performs a segmentation evaluation between predicted and target label maps.

    A table with per-subject stats will be included in the output dictionary
    under the key ``'subject_stats'``. The table is a ``pd.DataFrame`` where the
    first column header is ``"subject_name"`` and the other columns headers
    are ``stat_name.label_name``.

    A dictionary containing summary statistics will be included in the output dictionary
    under the key ``'summary_stats'``. It has the following structure:
    ``{summary_stat_name: {stat_name: {label_name: value}}}``

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
        prediction_label_map_name: Key to a ``tio.LabelMap`` in each ``tio.Subject`` that is evaluated.
            Typically this should point to the output of a segmentation model.
        target_label_map_name: Key to a ``tio.LabelMap`` in each ``tio.Subject`` that is evaluated.
            Typically this should point to the ground truth manually traced label.
        stats_to_output: A sequence of statistic names that are output from the evaluation
            If ``None`` then all statistics will be output.
        summary_stats_to_output: A sequence of summary statistic names that are output from the evaluation.
            If ``None`` then all summary statistics will be output.
    """
    def __init__(
            self,
            prediction_label_map_name: str,
            target_label_map_name: str,
            stats_to_output: Sequence[str] = None,
            summary_stats_to_output: Sequence[str] = None,
    ):
        self.prediction_label_map_name = prediction_label_map_name
        self.target_label_map_name = target_label_map_name
        self.stats_to_output = stats_to_output
        self.summary_stats_to_output = summary_stats_to_output

    def __call__(self, subjects):
        label_values = subjects[0][self.prediction_label_map_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject['name'] for subject in subjects]

        # Transform and stack prediction and target labels into one-hot (N, C, W, H, D) tensors
        prediction_label = torch.stack([
            subject[self.prediction_label_map_name].data for subject in subjects
        ]).bool()
        target_label = torch.stack([
            subject[self.target_label_map_name].data for subject in subjects
        ]).bool()

        # Compute (N, C) tensors for each statistic. Each element corresponds to one subject and one label.
        spatial_dims = (2, 3, 4)
        TP = (target_label & prediction_label).sum(dim=spatial_dims)
        FP = (~target_label & prediction_label).sum(dim=spatial_dims)
        TN = (~target_label & ~prediction_label).sum(dim=spatial_dims)
        FN = (target_label & ~prediction_label).sum(dim=spatial_dims)

        stats = {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'dice': 2 * TP / (2 * TP + FP + FN),
            'jaccard': TP / (TP + FP + FN),
            'sensitivity': TP / (TP + FN),
            'specificity': TN / (TN + FP),
            'precision': TP / (TP + FP),
            'recall': TP / (TP + FN)
        }

        out_dict = {}
        summary_stat_funcs = self.get_summary_stat_funcs()

        stats_to_output = self.stats_to_output
        if stats_to_output is None:
            stats_to_output = list(stats.keys())
        summary_stats_to_output = self.summary_stats_to_output
        if summary_stats_to_output is None:
            summary_stats_to_output = list(summary_stat_funcs.keys())

        # Convert the stats dict to a summary stats dict,
        # i.e. apply the summary stat functions on the subject axis for each (N, C) stat tensor,
        # Recall that the desired form is
        # {summary_stat_name: {stat_name: {label_name: value}}}
        out_dict['summary_stats'] = {
            summary_stat_name: {
                stat_name: {
                    label_name: func(stat[:, label_value].float()).tolist()
                    for label_name, label_value in label_values.items()
                }
                for stat_name, stat in stats.items()
                if stat_name in stats_to_output
            }
            for summary_stat_name, func in summary_stat_funcs.items()
            if summary_stat_name in summary_stats_to_output
        }

        # df is a pd.DataFrame with col=stat_name row_index=subject_name
        df = pd.DataFrame()
        df['subject'] = subject_names

        for label_name, label_value in label_values.items():
            for stat_name in stats_to_output:
                df[f'{stat_name}.{label_name}'] = stats[stat_name][:, label_value]

        out_dict['subject_stats'] = df

        return out_dict
