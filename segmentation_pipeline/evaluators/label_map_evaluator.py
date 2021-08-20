from typing import Sequence, Dict, Union
import numpy as np

from .evaluator import Evaluator
from .labeled_tensor import LabeledTensor


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
        curve_params: Dictionary with keys for label_names and values that contain params for curve
        curve_attribute: Subject key attribute that is passed into the curve, an example is 'age'.
        stats_to_output: A sequence of statistic names that are output from the evaluation
        summary_stats_to_output: A sequence of summary statistic names that are output from the evaluation.
    """
    def __init__(
            self,
            label_map_name: str,
            curve_params: Union[Dict[str, np.ndarray], None] = None,
            curve_attribute: Union[str, None] = None,
            stats_to_output: Sequence[str] = ('volume',),
            summary_stats_to_output: Sequence[str] = ('mean', 'std', 'min', 'max'),
    ):
        self.label_map_name = label_map_name
        self.curve_params = curve_params
        self.curve_attribute = curve_attribute
        self.stats_to_output = stats_to_output
        self.summary_stats_to_output = summary_stats_to_output

        # arguments check
        curve_stats = ['error', 'absolute_error', 'squared_error', 'percent_diff']
        if any(stat in curve_stats for stat in self.stats_to_output):
            if curve_params is None:
                raise ValueError("curve_params must be provided")

            if curve_attribute is None:
                raise ValueError("curve_attribute must be provided")


        if curve_params is not None and curve_attribute is not None:
            self.poly_func = {label: np.poly1d(param) for label, param in curve_params.items()}
        else:
            self.poly_func = None

    def __call__(self, subjects):
        label_values = subjects[0][self.label_map_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject['name'] for subject in subjects]

        subject_stats = LabeledTensor(dim_names=['subject', 'label', 'stat'],
                                      dim_keys=[subject_names, label_names, self.stats_to_output])
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

                if self.poly_func is not None:

                    curve_predicted = self.poly_func[label_name](subject[self.curve_attribute])
                    error = volume - curve_predicted
                    curve_stats = {
                        'error': error, 
                        'absolute_error': abs(error), 
                        'squared_error': error**2,
                        'percent_diff': (error / curve_predicted) * 100
                    }
                    stats.update(curve_stats)

                for stat_name in self.stats_to_output:
                    value = stats[stat_name].item()
                    subject_stats[subject['name'], label_name, stat_name] = value

        summary_stats = subject_stats.compute_summary_stats(self.summary_stats_to_output)
        out_dict = {
            'subject_stats': subject_stats.to_dataframe(),
            'summary_stats': summary_stats
        }

        return out_dict
