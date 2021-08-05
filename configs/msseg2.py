import os

import torchio as tio
from torch.optim import Adam, SGD
import torch
from torch.utils.data.sampler import SequentialSampler

from torch_context import TorchContext
from segmentation_trainer import SegmentationTrainer, ScheduledEvaluation
from models import *
from evaluation import HybridLogisticDiceLoss
from transforms import *
from data_processing import *
from evaluators import *
from predictors import *
from dataLoaderFactory import *
from utils import dont_collate


def get_context(device, variables, fold=0, **kwargs):
    context = TorchContext(device, name="msseg2", variables=variables)
    context.file_paths.append(os.path.abspath(__file__))
    context.config = config = {
        'fold': fold,
        'patch_size': 96
    }

    input_images = ["flair_time01", "flair_time02"]

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
    cohorts['all'] = RequireAttributes(input_images)
    cohorts['validation'] = RandomFoldFilter(num_folds=5, selection=fold, seed=0xDEADBEEF)
    cohorts['training'] = NegateFilter(cohorts['validation'])

    common_transforms_1 = tio.Compose([
        SetDataType(torch.float),
        EnforceConsistentAffine(source_image_name='flair_time01'),
        TargetResample(target_spacing=1, tolerance=0.11),
        CropToMask('brain_mask'),
        MinSizePad(config['patch_size'])
    ])

    augmentations = tio.Compose([
        RandomPermuteDimensions(),
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.OneOf({
            tio.RandomElasticDeformation(): 0.2,
            tio.RandomAffine(scales=0.2, degrees=45, default_pad_value='otsu'): 0.8,
        }, p=0.75),
        tio.RandomBiasField(p=0.5),
        tio.RescaleIntensity((0, 1), (0.01, 99.9)),
        tio.RandomGamma(p=0.8),
        tio.RescaleIntensity((-1, 1)),
        tio.RandomBlur((0, 1), p=0.2),
        tio.RandomNoise(std=0.1, p=0.35)
    ])

    common_transforms_2 = tio.Compose([
        tio.RescaleIntensity((-1, 1.), (0.05, 99.5)),
        ConcatenateImages(image_names=["flair_time01", "flair_time02"], image_channels=[1, 1], new_image_name="X"),
        RenameProperty(old_name='ground_truth', new_name='y'),
        CustomOneHot(include="y"),
    ])

    transforms = {
        'default': tio.Compose([
            common_transforms_1,
            common_transforms_2
        ]),
        'training': tio.Compose([
            common_transforms_1,
            augmentations,
            common_transforms_2,
            ImageFromLabels(
                new_image_name="patch_probability",
                label_weights=[('brain_mask', 'brain', 1), ('y', 'lesion', 100)]
            )
        ]),
    }

    context.add_component("dataset", SubjectFolder, root='$DATASET_PATH', subject_path="",
                          subject_loader=subject_loader, cohorts=cohorts, transforms=transforms)
    context.add_component("model", ModularUNet,
                          in_channels=2,
                          out_channels=2,
                          filters=[40, 40, 80, 80, 120, 120],
                          depth=6,
                          block_params={'residual': True},
                          downsample_class=BlurConv3d,
                          downsample_params={'kernel_size': 3, 'stride': 2, 'padding': 1},
                          upsample_class=BlurConvTranspose3d,
                          upsample_params={'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 0})
    context.add_component("optimizer", SGD, params="self.model.parameters()", lr=0.001, momentum=0.95)
    context.add_component("criterion", HybridLogisticDiceLoss, logistic_class_weights=[1, 100])

    training_evaluators = [
        ScheduledEvaluation(evaluator=SegmentationEvaluator('y_pred_eval', 'y_eval'),
                            log_name='training_segmentation_eval',
                            interval=15),
        ScheduledEvaluation(evaluator=LabelMapEvaluator('y_pred_eval'),
                            log_name='training_label_eval',
                            interval=15),
        ScheduledEvaluation(evaluator=ContourImageEvaluator("random", 'flair_time02', 'y_pred_eval', 'y_eval',
                                                            slice_id=0, legend=True, ncol=2, interesting_slice=True,
                                                            split_subjects=False),
                            log_name=f"contour_image",
                            interval=15),
    ]

    validation_evaluators = [
        ScheduledEvaluation(evaluator=SegmentationEvaluator("y_pred_eval", "y_eval"),
                            log_name="segmentation_eval",
                            cohorts=["validation"],
                            interval=50),
        ScheduledEvaluation(evaluator=ContourImageEvaluator("interesting", 'flair_time02', 'y_pred_eval', 'y_eval',
                                                            slice_id=0, legend=True, ncol=1, interesting_slice=True,
                                                            split_subjects=True),
                            log_name=f"contour_image",
                            cohorts=["validation"],
                            interval=50),
    ]

    def scoring_function(evaluation_dict):
        # Grab the output of the SegmentationEvaluator
        seg_eval = evaluation_dict['segmentation_eval']['validation']

        # Take mean dice, while accounting for subjects which have no lesions.
        # Dice is 0/0 = nan when the model correctly outputs no lesions. This is counted as a score of 1.0.
        # Dice is (>0)/0 = posinf when the model incorrectly predicts lesions when there are none.
        # This is counted as a score of 0.0.
        dice = torch.tensor(seg_eval["subject_stats"]['dice.lesion'])
        dice = dice.nan_to_num(nan=1.0, posinf=0.0)
        score = dice.mean()

        return score

    train_predictor = StandardPredict(device, image_names=['X', 'y'])
    val_predictor = PatchPredict(
        device=device,
        patch_batch_size=32,
        patch_size=config['patch_size'], 
        patch_overlap=(config['patch_size'] // 8),
        padding_mode=None,
        overlap_mode='average',
        image_names=['X']
    )

    patch_sampler = tio.WeightedSampler(patch_size=config['patch_size'], probability_map='patch_probability')
    train_dataloader_factory = PatchDataLoader(max_length=100, samples_per_volume=1, sampler=patch_sampler, collate_fn=dont_collate)
    val_dataloader_factory = StandardDataLoader(sampler=SequentialSampler, collate_fn=dont_collate)

    context.add_component("trainer", SegmentationTrainer,
                          training_batch_size=4,
                          save_rate=100,
                          scoring_interval=50,
                          scoring_function=scoring_function,
                          one_time_evaluators=[],
                          training_evaluators=training_evaluators,
                          validation_evaluators=validation_evaluators,
                          max_iterations_with_no_improvement=2000,
                          train_predictor=train_predictor,
                          val_predictor=val_predictor,
                          train_dataloader_factory=train_dataloader_factory,
                          val_dataloader_factory=val_dataloader_factory)

    return context
