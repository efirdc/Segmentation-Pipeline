from typing import Sequence

import torch
import torchio as tio
import pandas as pd

from transforms import *
from .evaluator import Evaluator


class LabelMapEvaluator(Evaluator):
    """ Computes statistics related to volume and shape of the structures in a label map.

    A table with per-subject stats will be included in the output dictionary
    under the key ``'subject_stats'``. The table is a ``pd.DataFrame`` where the
    first column header is ``"subject_name"`` and the other columns headers
    are ``"stat_name.label_name"``.

    A dictionary containing summary statistics will be included in the output dictionary
    under the key ``'summary_stats'``. It has the following structure:
    ``{summary_stat_name: {stat_name: {label_name: value}}}``

    The label map must have a ``Dict[str, int]`` property ``'label_values'``,
    which maps a label's name to its value.

    The supported output statistics are the following:
    ``('volume')``

    Summary statistics for any of the above can also be computed and output from the evaluation
    The following are supported:
    ``('mean', 'median', 'mode', 'std', 'min', 'max')``

    Args:
        label_map_name: Key to a ``tio.LabelMap`` in each ``tio.Subject``.
        stats_to_output: A sequence of statistic names that are output from the evaluation
            If ``None`` then all statistics will be output.
        summary_stats_to_output: A sequence of summary statistic names that are output from the evaluation.
            If ``None`` then all summary statistics will be output.
    """
    def __init__(
            self,
            label_map_name: str,
            stats_to_output: Sequence[str] = None,
            summary_stats_to_output: Sequence[str] = None,
    ):
        self.label_map_name = label_map_name
        self.stats_to_output = stats_to_output
        self.summary_stats_to_output = summary_stats_to_output

    def __call__(self, subjects):
        # TODO: Lots of repeated code from segmentation_evaluator.py, generalize this
        label_values = subjects[0][self.label_map_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject['name'] for subject in subjects]

        label = torch.stack([
            subject[self.label_map_name].data for subject in subjects
        ])

        spatial_dims = (-1, -2, -3)
        volume = label.sum(dim=spatial_dims)
        stats = {'volume': volume}

        out_dict = {}
        summary_stat_funcs = self.get_summary_stat_funcs()

        stats_to_output = self.stats_to_output
        if stats_to_output is None:
            stats_to_output = list(stats.keys())
        summary_stats_to_output = self.summary_stats_to_output
        if summary_stats_to_output is None:
            summary_stats_to_output = list(summary_stat_funcs.keys())

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

        df = pd.DataFrame()
        df['subject'] = subject_names

        for i, label_name in enumerate(label_values.keys()):
            for stat_name in stats_to_output:
                df[f'{stat_name}.{label_name}'] = stats[stat_name][:, i]
        out_dict['subject_stats'] = df

        return out_dict
