from typing import Sequence

from .evaluator import Evaluator
from .evaluation_dict import EvaluationDict


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
        summary_stats_to_output: A sequence of summary statistic names that are output from the evaluation.
    """
    def __init__(
            self,
            label_map_name: str,
            stats_to_output: Sequence[str] = ('volume',),
            summary_stats_to_output: Sequence[str] = ('mean', 'std', 'min', 'max'),
    ):
        self.label_map_name = label_map_name
        self.stats_to_output = stats_to_output
        self.summary_stats_to_output = summary_stats_to_output

    def __call__(self, subjects):
        label_values = subjects[0][self.label_map_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject['name'] for subject in subjects]

        subject_stats = EvaluationDict(dimensions=['subject', 'label', 'stat'],
                                       dimension_keys=[subject_names, label_names, self.stats_to_output])
        for subject in subjects:
            data = subject[self.label_map_name].data

            for label_name, label_value in label_values.items():
                label = (data == label_value)

                # Compute tensors for each statistic. Each element corresponds to one label.
                spatial_dims = (1, 2, 3)
                volume = label.sum(dim=spatial_dims)

                stats = {
                    'volume': volume,
                }

                for stat_name in self.stats_to_output:
                    value = stats[stat_name].item()
                    subject_stats[subject['name'], label_name, stat_name] = value

        summary_stats = subject_stats.compute_summary_stats(self.summary_stats_to_output)
        out_dict = {
            'subject_stats': subject_stats.to_dataframe(),
            'summary_stats': summary_stats
        }

        return out_dict
