import os

import torchio as tio
from torch.optim import Adam, SGD
import torch

from torch_context import TorchContext
from segmentation_trainer import SegmentationTrainer, ScheduledEvaluation
from models import NestedResUNet
from evaluation import HybridLogisticDiceLoss
from transforms import *
from data_processing import *
from evaluators import *


def get_context(device, variables, fold=0, **kwargs):
    context = TorchContext(device, name="msseg2", variables=variables)
    context.file_paths.append(os.path.abspath(__file__))
    context.config = config = {
        'fold': fold,
        'patch_size': 96
    }

    input_images = ["flair_time01", "flair_time02"]
    output_labels = ["ground_truth"]

    subject_loader = ComposeLoaders([
        ImageLoader(glob_pattern="flair_time01*", image_name='flair_time01', image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="flair_time02*", image_name='flair_time02', image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="brain_mask.*", image_name='brain_mask', image_constructor=tio.LabelMap,
                    label_values={"brain": 1}),
        ImageLoader(glob_pattern="ground_truth.*", image_name="ground_truth", image_constructor=tio.LabelMap,
                    label_values={"lesion": 1}
                    ),
    ])

    cohorts = {}
    cohorts['all'] = RequireAttributes(input_images + output_labels)
    cohorts['validation'] = RandomFoldFilter(num_folds=5, selection=fold, seed=0xDEADBEEF)
    cohorts['training'] = NegateFilter(cohorts['validation'])

    common_transforms_1 = tio.Compose([
        SetDataType(torch.float),
        EnforceConsistentAffine(source_image_name='flair_time01'),
    ])

    common_transforms_2 = tio.Compose([
        CropToMask('brain_mask'),
        tio.RescaleIntensity((-1, 1.), (0.0, 99.5)),
        ConcatenateImages(image_names=["flair_time01", "flair_time02"], image_channels=[1, 1], new_image_name="X"),
        RenameProperty(old_name='ground_truth', new_name='y'),
        CustomOneHot(include="y"),
    ])

    isotropic_params = {'tolerance': 0.15, 'min_spacing': 0.85}
    transforms = {
        'default': tio.Compose([
            common_transforms_1,
            IsotropicResample(spacing_mode='median', **isotropic_params),
            common_transforms_2
        ]),
        'training': tio.Compose([
            common_transforms_1,
            tio.OneOf({
                IsotropicResample(spacing_mode='median', **isotropic_params): 0.5,
                IsotropicResample(spacing_mode='min', **isotropic_params): 0.25,
                IsotropicResample(spacing_mode='max', **isotropic_params): 0.25,
            }),
            common_transforms_2,
            ImageFromLabels(
                new_image_name="patch_probability",
                label_weights=[('brain_mask', 'brain', 1), ('y', 'lesion', 100)]
            )
        ]),
    }

    context.add_component("dataset", SubjectFolder, root='$DATASET_PATH', subject_path="",
                          subject_loader=subject_loader, cohorts=cohorts, transforms=transforms)
    context.add_component("model", NestedResUNet, input_channels=2, output_channels=2,
                          filters=20, dropout_p=0.2, saggital_split=False)
    context.add_component("optimizer", SGD, params="self.model.parameters()", lr=0.01, momentum=0.95)
    context.add_component("criterion", HybridLogisticDiceLoss)

    training_evaluators = [
        ScheduledEvaluation(evaluator=SegmentationEvaluator('y_pred_eval', 'y_eval'),
                            log_name='training_segmentation_eval'),
        ScheduledEvaluation(evaluator=LabelMapEvaluator('y_pred_eval'),
                            log_name='training_label_eval'),
        ScheduledEvaluation(evaluator=ContourImageEvaluator("random", 'flair_time02', 'y_pred_eval', 'y_eval',
                                                            slice_id=0, legend=True, ncol=2, interesting_slice=True,
                                                            split_subjects=False),
                            log_name=f"contour_image",
                            interval=1),
    ]

    validation_evaluators = [
        ScheduledEvaluation(evaluator=SegmentationEvaluator("y_pred_eval", "y_eval"),
                            log_name="segmentation_eval",
                            cohorts=["validation"],
                            interval=25),
        ScheduledEvaluation(evaluator=ContourImageEvaluator("interesting", 'flair_time02', 'y_pred_eval', 'y_eval',
                                                            slice_id=0, legend=True, ncol=1, interesting_slice=True,
                                                            split_subjects=True),
                            log_name=f"contour_image",
                            cohorts=["validation"],
                            interval=10),
    ]

    def scoring_function(evaluation_dict):
        # Grab the output of the SegmentationEvaluator
        seg_eval = evaluation_dict['segmentation_eval']['validation']

        # Take mean dice, while accounting for subjects which have no lesions.
        # Dice is 0/0 = nan when the model correctly outputs no lesions. This is counted as a score of 1.0.
        # Dice is (>0)/0 = inf when the model incorrectly predicts lesions when there are none.
        # This is counted as a score of 0.0.
        dice = torch.tensor(seg_eval["subject_stats"]['dice.lesion'])
        dice.nan_to_num(nan=1.0, posinf=0.0)
        score = dice.mean()

        return score

    patch_sampler = tio.WeightedSampler(patch_size=config['patch_size'], probability_map='patch_probability')
    context.add_component("trainer", SegmentationTrainer,
                          training_batch_size=4,
                          save_rate=100,
                          scoring_interval=50,
                          scoring_function=scoring_function,
                          one_time_evaluators=[],
                          training_evaluators=training_evaluators,
                          validation_evaluators=validation_evaluators,
                          max_iterations_with_no_improvement=500,
                          enable_patch_mode=True,
                          patch_size=config['patch_size'],
                          training_patch_sampler=patch_sampler,
                          training_patches_per_volume=1,
                          inference_patch_overlap=(config['patch_size'] // 8),
                          inference_padding_mode=None,
                          inference_overlap_mode='average')

    return context
