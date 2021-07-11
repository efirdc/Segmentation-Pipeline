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

        if self.stats_to_output is None:
            self.stats_to_output = ('volume',)
        if self.summary_stats_to_output is None:
            self.summary_stats_to_output = ('mean', 'median', 'mode', 'std', 'min', 'max')

    def __call__(self, subjects):
        label_values = subjects[0][self.label_map_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject['name'] for subject in subjects]

        subject_stats = {
            subject_name: {
                label_name: None
                for label_name in label_names
            }
            for subject_name in subject_names
        }

        for subject in subjects:
            label_data = subject[self.label_map_name].data.bool()

            spatial_dims = (1, 2, 3)
            stats = {
                'volume': label_data.sum(dim=spatial_dims),
            }

            for label_name, label_value in label_values.items():
                subject_stats[subject['name']][label_name] = {
                    stat_name: stat_value[label_value]
                    for stat_name, stat_value in stats.items()
                }

        # Recall that the desired form is
        # {summary_stat_name: {stat_name: {label_name: value}}}
        summary_stats = {
            summary_stat_name: {
                stat_name: {
                    label_name: None for label_name in label_names
                }
                for stat_name in self.stats_to_output
            }
            for summary_stat_name in self.summary_stats_to_output
        }

        df = pd.DataFrame()
        df['subject'] = subject_names

        summary_stat_funcs = self.get_summary_stat_funcs()

        for summary_stat_name in self.summary_stats_to_output:
            for stat_name in self.stats_to_output:
                for label_name in label_names:
                    subject_values = torch.tensor([
                        subject_stats[subject_name][label_name][stat_name]
                        for subject_name in subject_names
                    ]).float()

                    summary_stat_func = summary_stat_funcs[summary_stat_name]
                    summary_stat = summary_stat_func(subject_values).float()
                    summary_stats[summary_stat_name][stat_name][label_name] = summary_stat

                    df[f'{stat_name}.{label_name}'] = subject_values

        out_dict = {'subject_stats': df, 'summary_stats': summary_stats}

        return out_dict
