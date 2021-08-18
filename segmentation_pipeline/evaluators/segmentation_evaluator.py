from typing import Sequence

from .evaluator import Evaluator
from .labeled_tensor import LabeledTensor


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
    ``('target_volume', 'prediction_volume', 'TP', 'FP', 'TN', 'FN',
       'dice', 'jaccard', 'precision', 'recall')``

    Summary statistics for any of the above can also be computed and output from the evaluation
    The following are supported:
    ``('mean', 'median', 'mode', 'std', 'min', 'max')``

    Args:
        prediction_label_map_name: Key to a ``tio.LabelMap`` in each ``tio.Subject`` that is evaluated.
            Typically this should point to the output of a segmentation model.
        target_label_map_name: Key to a ``tio.LabelMap`` in each ``tio.Subject`` that is evaluated.
            Typically this should point to the ground truth manually traced label.
        stats_to_output: A sequence of statistic names that are output from the evaluation
        summary_stats_to_output: A sequence of summary statistic names that are output from the evaluation.
    """
    def __init__(
            self,
            prediction_label_map_name: str,
            target_label_map_name: str,
            stats_to_output: Sequence[str] = ('target_volume', 'prediction_volume',
                                              'TP', 'FP', 'TN', 'FN', 'dice', 'precision', 'recall'),
            summary_stats_to_output: Sequence[str] = ('mean', 'std', 'min', 'max'),
    ):
        self.prediction_label_map_name = prediction_label_map_name
        self.target_label_map_name = target_label_map_name
        self.stats_to_output = stats_to_output
        self.summary_stats_to_output = summary_stats_to_output

    def __call__(self, subjects):
        label_values = subjects[0][self.prediction_label_map_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject['name'] for subject in subjects]

        subject_stats = LabeledTensor(dim_names=['subject', 'label', 'stat'],
                                      dim_keys=[subject_names, label_names, self.stats_to_output])

        for subject in subjects:
            pred_data = subject[self.prediction_label_map_name].data
            target_data = subject[self.target_label_map_name].data

            for label_name, label_value in label_values.items():
                pred_label = (pred_data == label_value)
                target_label = (target_data == label_value)

                # Compute tensors for each statistic. Each element corresponds to one label.
                spatial_dims = (1, 2, 3)
                TP = (target_label & pred_label).sum(dim=spatial_dims).float()
                FP = (~target_label & pred_label).sum(dim=spatial_dims).float()
                TN = (~target_label & ~pred_label).sum(dim=spatial_dims).float()
                FN = (target_label & ~pred_label).sum(dim=spatial_dims).float()

                stats = {
                    'target_volume': TP + FN,
                    'prediction_volume': TP + FP,
                    'TP': TP,
                    'FP': FP,
                    'TN': TN,
                    'FN': FN,
                    'dice': 2 * TP / (2 * TP + FP + FN),
                    'jaccard': TP / (TP + FP + FN),
                    'precision': TP / (TP + FP),
                    'recall': TP / (TP + FN),
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
