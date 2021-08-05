from typing import Sequence

from .evaluator import Evaluator
from .evaluation_dict import EvaluationDict


class ImageRegionEvaluator(Evaluator):
    """
    """
    def __init__(
            self,
            label_map_name: str,
            image_names: Sequence[str],
            stats_to_output: Sequence[str] = None,
            summary_stats_to_output: Sequence[str] = None,
    ):
        self.label_map_name = label_map_name
        self.image_names = image_names
        self.stats_to_output = stats_to_output
        self.summary_stats_to_output = summary_stats_to_output

    def __call__(self, subjects):
        label_values = subjects[0][self.label_map_name]['label_values']
        label_names = list(label_values.keys())
        subject_names = [subject['name'] for subject in subjects]

        subject_stats = EvaluationDict(dimensions=['subject', 'label', 'image_name', 'stat'],
                                       dimension_keys=[subject_names, label_names,
                                                       self.image_names, self.stats_to_output])

        # TODO: The point of this evaluator is to calculate a summary stats on an image inside the mask
        # of a particular label. However its kind of hard since images will always have normalizations
        # applied to them. Torchio intensity transforms should probably cache the shift/scale applied
        # to images so it can be inverted easily.

        raise NotImplementedError()
