from typing import Sequence

import torch
import torchio as tio

from transforms import *
from .evaluator import Evaluator

#TODO: Finish this

class LabelMapEvaluator(Evaluator):
    """ Computes statistics related to volume and shape of the structures in a label map.

    A table with per-subject stats will be included in the output dictionary
    under the key ``'subject_stats'``. The table is a ``pd.DataFrame`` where the
    first column header is ``"subject_name"`` and the other columns headers
    are ``stat_name.label_name``.

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
        output_summary_stats: If ``True`` then a dictionary containing summary statistics
            will be included in the output dictionary under the key ``'summary_stats'``.
            Has the structure ``{summary_stat_name: {stat_name: {label_name: value}}}``
        output_subject_stats: If ``True`` then a table with per-subject stats
             will be included in the output dictionary under the key ``'subject_stats'``.
             The table is a ``pd.DataFrame`` where the first column contains a ``'subject_name'``
             and the other columns are ``stat_name.label_name``.
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
            label_map_name: str,
            stats_to_output: Sequence[str] = None,
            summary_stats_to_output: Sequence[str] = None,
    ):
        self.label_map_name = label_map_name
        self.stats_to_output = stats_to_output
        self.summary_stats_to_output = summary_stats_to_output

    def __call__(self, subjects):
        label_values = subjects[0][self.label_map_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject['name'] for subject in subjects]
        summary_stat_funcs = self.get_summary_stat_funcs(dim=0)

        transform = tio.Compose([
            CustomSequentialLabels(),
            CustomOneHot(len(label_values))
        ])

        # Apply the transformation to each label map
        # label_dict format is {subject_name: label_mask} where label_mask is a (C, W, H, D) bool tensor
        label_dict = {
            subject['name']: transform(subject[self.label_map_name]).data.bool()
            for subject in subjects
        }

        # Split the masks on the channel axis so they correspond to only one label value
        # label_dict format is {subject_name: {label_name: {label_mask}} where label_mask is (1, W, H, D) bool tensor
        label_dict = {
            label_name: {
                subject_name: label[label_value:label_value + 1]
                for subject_name, label in label_dict.items()
            }
            for label_name, label_value in label_values.items()
        }

        # Get the volume stat for each (1, W, H, D) label_mask
        spatial_dims = (-1, -2, -3)
        volume = {
            label_name: {
                subject_name: {
                    label_dict[label_name][subject_name].sum(dim=spatial_dims)
                }
                for subject_name in subject_names
            }
            for label_name in label_names
        }

        if self.image_names:

            def apply_mask(subject, image_name, label_name):
                image = subject[image_name].data
                label_mask = label_dict[label_name][subject['name']]
                image, label_mask = torch.broadcast_tensors(image, label_mask)
                image_region = image[label_mask]
                return image_region

            # Apply the `label_mask`s on each image we are interested in to get `image_region`s
            # `image_region`s are 1D tensors containing voxel values of an image that are within a `label_mask`
            # Format {label_name: {image_name: {subject_name: image_region}}}
            image_regions = {
                label_name: {
                    image_name: {
                        subject['name']: apply_mask(subject, image_name, label_name)
                        for subject in subjects
                    }
                    for image_name in self.image_names
                }
                for label_name in label_names
            }

            # Apply summary stat functions on the 1D `image_region`s
            # Format {stat_name: {label_name: {image_name: {subject_name: value}}}}
            image_stats = {
                stat_name: {
                    label_name: {
                        image_name: {
                            subject_name: {
                                stat_func(image_regions[label_name][image_name][subject_name])
                            }
                            for subject_name in subject_names
                        }
                        for image_name in self.image_names
                    }
                    for label_name in label_names
                }
                for stat_name, stat_func in summary_stat_funcs.items()
                if stat_name in self.stats_to_output
            }

        if self.output_summary_stats:
            volume_summary_stat = {
                summary_stat_name: {
                    label_name: summary_stat_func(torch.tensor([
                        volume[label_name][subject_name]
                        for subject_name in subject_names
                    ]))
                    for label_name in label_names
                }
                for summary_stat_name, summary_stat_func in summary_stat_funcs.items()
                if summary_stat_name in self.summary_stats_to_output
            }

            if self.image_names:
                image_summary_stats = {
                    summary_stat_name: {
                        stat_name: {
                            label_name: {
                                image_name: summary_stat_func(torch.tensor([
                                    image_stats[stat_name][label_name][image_name][subject_name]
                                    for subject_name in subject_names
                                ]))
                                for image_name in self.image_names
                            }
                            for label_name in label_names
                        }
                        for stat_name in self.stats_to_output
                    }
                    for summary_stat_name, summary_stat_func in summary_stat_funcs.items()
                    if summary_stat_name in self.summary_stats_to_output
                }
