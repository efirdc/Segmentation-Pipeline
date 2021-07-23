from typing import Sequence, Dict, Any, Callable

import torch
from skimage.morphology import label

from .evaluator import Evaluator
from .evaluation_dict import EvaluationDict


def msseg_detection_test(
        overlap_histogram,
        min_recall=0.1,
        contribution_threshold=0.65,
        min_precision=0.3,
):
    """
    Detection test as described in "Objective Evaluation of Multiple Sclerosis Lesion Segmentation
    using a Data Management and Processing Infrastructure"

    This metric was used in the MSSEG (2016) and MSSEG2 (2021) lesion segmentation challenges.

    Denote N as the number of target instances and M the number of predicted instances.

    Returns a boolean tensor of length N that is True if the ith target instance is detected
    by the M predicted instances.

    Args:
        overlap_histogram: A (N + 1, M + 1) array. Element [i, j] is the number of overlapping voxels
            between the ith target component and the jth predicted component.
            Component 0 is the background for both prediction and target.
        min_recall: Target instances that are not overlapped by min_recall of predicted instances
            are classified as not detected. Equivalent to the alpha parameter in the paper.
        contribution_threshold: The contribution of each predicted instance to the target instance is
            sorted in descending order. The predicted instances which contribute to the first
            contribution_theshold of the overlap have an additional precision requirement.
            Equivalent to the gamma parameter in the paper.
        min_precision: Predicted instances that are within the contribution_threshold have an
            additional min_precision requirement. This is equivalent to (1 - beta) in the paper.
    """
    N = overlap_histogram.shape[0] - 1
    M = overlap_histogram.shape[1] - 1

    target_volume = overlap_histogram.sum(dim=1)
    prediction_volume = overlap_histogram.sum(dim=0)

    detected = []
    for i in range(1, N + 1):

        # Check if the target instance is not sufficiently overlapped by the predicted instances
        target_TP = overlap_histogram[i, 1:].sum()
        recall = target_TP / target_volume[i]
        if recall < min_recall:
            detected.append(False)
            continue

        # Indices which sort the predicted instances by how much they overlap the target instance
        predicted_ids = torch.argsort(overlap_histogram[i, 1:], descending=True) + 1

        # Check if predicted instances have the required precision until the contribution_threshold is reached
        contribution_total = 0.0
        for j in predicted_ids:
            precision = overlap_histogram[i, j] / prediction_volume[j]
            if precision < min_precision:
                detected.append(False)
                break
            contribution = overlap_histogram[i, j] / target_TP
            contribution_total += contribution
            if contribution_total >= contribution_threshold:
                detected.append(True)
                break

    return torch.tensor(detected)


class InstanceSegmentationEvaluator(Evaluator):
    """
    """
    def __init__(
            self,
            prediction_label_map_name: str,
            target_label_map_name: str,
            stats_to_output: Sequence[str] = ('target_components', 'predicted_components',
                                              'target_detections', 'predicted_detections',
                                              'detection_recall', 'detection_precision', 'detection_f1',
                                              'target_volume', 'prediction_volume', 'TP', 'FP', 'TN', 'FN',
                                              'dice', 'jaccard', 'precision', 'recall',),
            summary_stats_to_output: Sequence[str] = ('mean', 'std', 'min', 'max', 'median', 'mode'),
            connectivity: int = 2,
            detection_test: Callable = msseg_detection_test,
            detection_test_params: Dict[str, Any] = None,
    ):
        self.prediction_label_map_name = prediction_label_map_name
        self.target_label_map_name = target_label_map_name
        self.stats_to_output = stats_to_output
        self.summary_stats_to_output = summary_stats_to_output
        self.connectivity = connectivity
        self.detection_test = detection_test
        self.detection_test_params = {} if detection_test_params is None else detection_test_params

    def __call__(self, subjects):

        subject_names = [subject['name'] for subject in subjects]
        subject_stats = EvaluationDict(dimensions=['subject', 'stat'],
                                       dimension_keys=[subject_names, self.stats_to_output])

        for subject in subjects:
            pred_data = subject[self.prediction_label_map_name].data
            target_data = subject[self.target_label_map_name].data

            if pred_data.shape[0] != 2 or target_data.shape[0] != 2:
                raise RuntimeError("Instance segmentation evaluation is only supported for single class predictions.")

            label_params = {'return_num': True, 'connectivity': self.connectivity}
            pred_components, num_pred_components = label(pred_data[1].numpy(), **label_params)
            target_components, num_target_components = label(target_data[1].numpy(), **label_params)
            N, M = num_target_components, num_pred_components,

            pred_components = torch.from_numpy(pred_components)
            target_components = torch.from_numpy(target_components)

            #_, pred_volumes = torch.unique(pred_components, sorted=True, return_counts=True)
            #_, target_volumes = torch.unique(target_components, sorted=True, return_counts=True)

            # Trick to encode the overlap of target components i and predicted components j as (i + j * factor)
            # Then torch.unique can be used to count the occurrences of each overlap
            # Alternative would be to stack them and call unique on the flattened spatial dimensions,
            # however that is extremely slow while for some reason this approach is very fast.
            factor = 1000000
            overlap_components = target_components + (pred_components * factor)
            unique_overlap, overlap_counts = torch.unique(overlap_components, sorted=True, return_counts=True)

            overlap_histogram = torch.zeros(N + 1, M + 1)
            for overlap_component, count in zip(unique_overlap, overlap_counts):
                i = overlap_component % factor
                j = overlap_component // factor
                overlap_histogram[i, j] = count

            target_detected = self.detection_test(overlap_histogram, **self.detection_test_params)
            prediction_detected = self.detection_test(overlap_histogram.T, **self.detection_test_params)

            detection_recall = target_detected.sum() / N
            detection_precision = prediction_detected.sum() / M
            detection_f1 = 2 * (detection_recall * detection_precision) / (detection_recall + detection_precision)

            TP = overlap_histogram[1:, 1:].sum()
            FP = overlap_histogram[0, 1:].sum()
            TN = overlap_histogram[0, 0].sum()
            FN = overlap_histogram[1:, 0].sum()

            stats = {
                'target_components': N,
                'predicted_components': M,
                'target_detections': target_detected.sum(),
                'predicted_detections': prediction_detected.sum(),
                'detection_recall': detection_recall,
                'detection_precision': detection_precision,
                'detection_f1': detection_f1,
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
                value = stats[stat_name]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                subject_stats[subject['name'], stat_name] = value

        summary_stats = subject_stats.compute_summary_stats(self.summary_stats_to_output)
        out_dict = {
            'subject_stats': subject_stats.to_dataframe(),
            'summary_stats': summary_stats
        }

        return out_dict
