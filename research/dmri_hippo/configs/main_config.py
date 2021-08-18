import os

from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torchio as tio

from segmentation_pipeline import *


old_validation_split = [f"cbbrain_{subject_id:03}" for subject_id in (
    32, 42, 55, 67, 82, 88, 96, 98, 102, 107, 110, 117, 123, 143, 145, 149, 173, 182, 184, 401
)]


def get_context(
        device,
        variables,
        fold=0,
        predict_hbt=False,
        training_batch_size=4,
):
    context = TorchContext(device, name="dmri-hippo", variables=variables)
    context.file_paths.append(os.path.abspath(__file__))

    input_images = ["mean_dwi", "md", "fa"]
    output_labels = ["whole_roi", "hbt_roi"]

    subject_loader = ComposeLoaders([
        ImageLoader(glob_pattern="mean_dwi.*", image_name='mean_dwi', image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="md.*", image_name='md', image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="fa.*", image_name='fa', image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="full_dwi.*", image_name='full_dwi', image_constructor=tio.ScalarImage),
        TensorLoader(glob_pattern="full_dwi_grad.b", tensor_name="grad", belongs_to="full_dwi"),
        ImageLoader(glob_pattern="whole_roi.*", image_name="whole_roi", image_constructor=tio.LabelMap,
                    label_values={"left_whole": 1, "right_whole": 2}
                    ),
        ImageLoader(glob_pattern="hbt_roi.*", image_name="hbt_roi", image_constructor=tio.LabelMap,
                    label_values={"left_head": 1, "left_body": 2, "left_tail": 3, "right_head": 4, "right_body": 5,
                                 "right_tail": 6}
                    ),
        ImageLoader(glob_pattern="../../atlas/whole_roi_union.*", image_name="whole_roi_union",
                    image_constructor=tio.LabelMap, uniform=True),
        AttributeLoader(glob_pattern='attributes.*'),
        AttributeLoader(glob_pattern='../../attributes/cross_validation_split.json',
                        multi_subject=True, uniform=True),
        AttributeLoader(glob_pattern='../../attributes/ab300_validation_subjects.json',
                        multi_subject=True, uniform=True),
        AttributeLoader(glob_pattern='../../attributes/cbbrain_test_subjects.json',
                        multi_subject=True, uniform=True),
    ])

    cohorts = {}
    cohorts['all'] = RequireAttributes(input_images)
    cohorts['cross_validation'] = RequireAttributes(['fold'])
    cohorts['training'] = ComposeFilters([
        cohorts['cross_validation'], ForbidAttributes({"fold": fold})
    ])
    cohorts['cbbrain_validation'] = ComposeFilters([
        cohorts['cross_validation'], RequireAttributes({"fold": fold})
    ])
    cohorts['cbbrain_test'] = RequireAttributes({'cbbrain_test': True})
    cohorts['ab300_validation'] = RequireAttributes({'ab300_validation': True})
    cohorts['ab300_validation_plot'] = ComposeFilters([
        cohorts['ab300_validation'], RandomSelectFilter(num_subjects=20)
    ])
    cohorts['cbbrain'] = RequireAttributes({"protocol": "cbbrain"})
    cohorts['ab300'] = RequireAttributes({"protocol": "ab300"})
    cohorts['rescans'] = ForbidAttributes({"rescan_id": "None"})
    cohorts['fasd'] = RequireAttributes({"pathologies": "FASD"})

    common_transforms_1 = tio.Compose([
        tio.CropOrPad((96, 88, 24), padding_mode='minimum', mask_name='whole_roi_union'),
        MergeLabels([('left_whole', 'right_whole')], right_masking_method="Right", include="whole_roi"),
        MergeLabels([('left_head', 'right_head'), ('left_body', 'right_body'), ('left_tail', 'right_tail')],
                    right_masking_method="Right", include="hbt_roi"),
    ])

    noise = tio.RandomNoise(std=0.035, p=0.3)
    blur = tio.RandomBlur((0, 1), p=0.2)
    standard_augmentations = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.RandomElasticDeformation(p=0.5, num_control_points=(7, 7, 4), locked_borders=1,
                                     image_interpolation='bspline', exclude="full_dwi"),
        tio.RandomBiasField(p=0.5),
        tio.RescaleIntensity((0, 1), (0.01, 99.9)),
        tio.RandomGamma(p=0.8),
        tio.RescaleIntensity((-1, 1)),
        tio.OneOf([
            tio.Compose([blur, noise]),
            tio.Compose([noise, blur]),
        ])
    ], exclude="full_dwi")
    dwi_augmentation = ReconstructMeanDWI(num_dwis=(1, 25), num_directions=(1, 3), directionality=(4, 10))

    common_transforms_2 = tio.Compose([
        tio.RescaleIntensity((-1., 1.), (0.5, 99.5)),
        ConcatenateImages(image_names=["mean_dwi", "md", "fa"], image_channels=[1, 1, 1], new_image_name="X"),
        RenameProperty(old_name="hbt_roi" if predict_hbt else "whole_roi", new_name="y"),
        CustomOneHot(include="y")
    ])

    transforms = {
        'default': tio.Compose([
            common_transforms_1,
            common_transforms_2
        ]),
        'training': tio.Compose([
            common_transforms_1,
            tio.Compose([dwi_augmentation, standard_augmentations]),
            common_transforms_2
        ]),
    }

    context.add_component("dataset", SubjectFolder, root='$DATASET_PATH', subject_path="subjects",
                          subject_loader=subject_loader, cohorts=cohorts, transforms=transforms)
    context.add_component("model", NestedResUNet,
                          input_channels=3,
                          output_channels=4 if predict_hbt else 2,
                          filters=40,
                          dropout_p=0.2)
    context.add_component("optimizer", Adam, params="self.model.parameters()", lr=0.0002)
    context.add_component("criterion", HybridLogisticDiceLoss)

    training_evaluators = [
        ScheduledEvaluation(evaluator=SegmentationEvaluator('y_pred_eval', 'y_eval'),
                            log_name='training_segmentation_eval',
                            interval=10),
        ScheduledEvaluation(evaluator=ContourImageEvaluator("Axial", 'mean_dwi', 'y_pred_eval', 'y_eval',
                                                            slice_id=12, legend=True, ncol=2, split_subjects=False),
                            log_name=f"contour_image_training",
                            interval=10),
    ]

    validation_evaluators = [
        ScheduledEvaluation(evaluator=LabelMapEvaluator('y_pred_eval'),
                            log_name="predicted_label_eval",
                            cohorts=['cbbrain_validation', 'ab300_validation'],
                            interval=50),
        ScheduledEvaluation(evaluator=SegmentationEvaluator("y_pred_eval", "y_eval"),
                            log_name="segmentation_eval",
                            cohorts=['cbbrain_validation'],
                            interval=50),
        ScheduledEvaluation(evaluator=ContourImageEvaluator("Axial", "mean_dwi", "y_pred_eval", "y_eval",
                                                            slice_id=10, legend=True, ncol=5, split_subjects=False),
                            log_name="contour_image_axial",
                            cohorts=['cbbrain_validation', 'ab300_validation_plot'],
                            interval=25),
        ScheduledEvaluation(evaluator=ContourImageEvaluator("Coronal", "mean_dwi", "y_pred_eval", "y_eval",
                                                            slice_id=44, legend=True, ncol=2, split_subjects=False),
                            log_name="contour_image_coronal",
                            cohorts=['cbbrain_validation', 'ab300_validation_plot'],
                            interval=25),
    ]

    def scoring_function(evaluation_dict):
        # Grab the output of the SegmentationEvaluator
        seg_eval_cbbrain = evaluation_dict['segmentation_eval']['cbbrain_validation']["summary_stats"]

        # Get the mean dice for each label (the mean is across subjects)
        cbbrain_dice = seg_eval_cbbrain['mean', :, 'dice']

        # Now take the mean across all labels
        cbbrain_dice = cbbrain_dice.mean()
        score = cbbrain_dice
        return score

    train_predictor = StandardPredict(sagittal_split=True, image_names=['X', 'y'])
    validation_predictor = StandardPredict(sagittal_split=True, image_names=['X'])

    train_dataloader_factory = StandardDataLoader(sampler=RandomSampler)
    validation_dataloader_factory = StandardDataLoader(sampler=SequentialSampler)

    context.add_component("trainer", SegmentationTrainer,
                          training_batch_size=training_batch_size,
                          save_rate=100,
                          scoring_interval=50,
                          scoring_function=scoring_function,
                          one_time_evaluators=[],
                          training_evaluators=training_evaluators,
                          validation_evaluators=validation_evaluators,
                          max_iterations_with_no_improvement=500,                           
                          train_predictor=train_predictor,
                          validation_predictor=validation_predictor,
                          train_dataloader_factory=train_dataloader_factory,
                          validation_dataloader_factory=validation_dataloader_factory)

    return context
