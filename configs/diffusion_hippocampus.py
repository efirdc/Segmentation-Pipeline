import os

from torch.utils.data.sampler import RandomSampler, SequentialSampler

from torch_context import TorchContext
import torchio as tio
from segmentation_trainer import SegmentationTrainer, ScheduledEvaluation
from models import NestedResUNet
from evaluation import HybridLogisticDiceLoss
from torch.optim import Adam

from transforms import *
from data_processing import *
from evaluators import *
from predictors import *
from data_loader_factory import *


def get_context(device, variables, predict_hbt=False, **kwargs):
    context = TorchContext(device, name="dmri-hippo", variables=variables)
    context.file_paths.append(os.path.abspath(__file__))

    input_images = ["mean_dwi", "md", "fa"]
    output_labels = ["whole_roi", "hbt_roi"]

    subject_loader = ComposeLoaders([
        ImageLoader(glob_pattern="mean_dwi.*", image_name='mean_dwi', image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="md.*", image_name='md', image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="fa.*", image_name='fa', image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="whole_roi.*", image_name="whole_roi", image_constructor=tio.LabelMap,
                    label_values={"left_whole": 1, "right_whole": 2}
                    ),
        ImageLoader(glob_pattern="hbt_roi.*", image_name="hbt_roi", image_constructor=tio.LabelMap,
                    label_values={"left_head": 1, "left_body": 2, "left_tail": 3, "right_head": 4, "right_body": 5,
                                 "right_tail": 6}
                    ),
        AttributeLoader(glob_pattern='attributes.*')
    ])

    cohorts = {}
    cbbrain_validation_subjects = [f"cbbrain_{subject_id:03}" for subject_id in (
        32, 42, 55, 67, 82, 88, 96, 98, 102, 107, 110, 117, 123, 143, 145, 149, 173, 182, 184, 401
    )]
    cohorts['all'] = RequireAttributes(input_images)
    cohorts['training'] = ComposeFilters([
        RequireAttributes(output_labels),
        RequireAttributes({"pathologies": "None", "protocol": "cbbrain", "rescan_id": "None"}),
        ForbidAttributes({"name": cbbrain_validation_subjects})
    ])
    cohorts['cbbrain'] = RequireAttributes({"protocol": "cbbrain"})
    cohorts['ab300'] = RequireAttributes({"protocol": "ab300"})
    cohorts['rescans'] = ForbidAttributes({"rescan_id": "None"})
    cohorts['labeled'] = RequireAttributes(['y'])
    cohorts['cbbrain_validation'] = RequireAttributes({"name": cbbrain_validation_subjects})
    cohorts['ab300_validation'] = ComposeFilters([cohorts['ab300'], cohorts['labeled'], RequireAttributes({"rescan_id": "None"})])
    cohorts['fasd'] = ComposeFilters([cohorts['labeled'], RequireAttributes({"pathologies": "FASD"})])

    common_transforms = tio.Compose([
        tio.Crop((62, 62, 70, 58, 0, 0)),
        tio.Pad((0, 0, 0, 0, 2, 2), padding_mode="minimum"),
        MergeLabels([('left_whole', 'right_whole')], right_masking_method="Right", include="whole_roi"),
        MergeLabels([('left_head', 'right_head'), ('left_body', 'right_body'), ('left_tail', 'right_tail')],
                    right_masking_method="Right", include="hbt_roi"),
        ConcatenateImages(image_names=["mean_dwi", "md", "fa"], image_channels=[1, 1, 1], new_image_name="X"),
        RenameProperty(old_name="hbt_roi" if predict_hbt else "whole_roi", new_name="y"),
        CustomOneHot(include="y")
    ])

    transforms = {
        'default': tio.Compose([
            tio.RescaleIntensity((-1, 1.), (0.5, 99.5)),
            common_transforms
        ]),
        'training': tio.Compose([
            tio.RescaleIntensity((0, 1.), (0.5, 99.5)),
            tio.RandomBiasField(coefficients=0.5, include=["mean_dwi"]),
            tio.RescaleIntensity((-1, 1), (0., 99.5)),
            common_transforms
        ]),
    }

    context.add_component("dataset", SubjectFolder, root='$DATASET_PATH', subject_path="subjects",
                          subject_loader=subject_loader, cohorts=cohorts, transforms=transforms)
    context.add_component("model", NestedResUNet, input_channels=3, output_channels=4 if predict_hbt else 2,
                          filters=40, dropout_p=0.2, saggital_split=True)
    context.add_component("optimizer", Adam, params="self.model.parameters()", lr=0.0002)
    context.add_component("criterion", HybridLogisticDiceLoss)

    plot_subjects = ["cbbrain_042", "cbbrain_082", "cbbrain_143",
                     "cbbrain_036", "cbbrain_039", "cbbrain_190",  # FASD
                     "ab300_002", "ab300_005", "ab300_090"]

    one_time_evaluators = [
        ScheduledEvaluation(evaluator=LabelMapEvaluator('y_eval'),
                            log_name="ground_truth_label_eval",
                            cohorts=['labeled']),
    ]

    training_evaluators = [
        ScheduledEvaluation(evaluator=SegmentationEvaluator('y_pred_eval', 'y_eval'),
                            log_name='training_segmentation_eval'),
        ScheduledEvaluation(evaluator=LabelMapEvaluator('y_pred_eval'),
                            log_name='training_label_eval'),
    ]

    validation_evaluators = [
        ScheduledEvaluation(evaluator=LabelMapEvaluator('y_pred_eval'),
                            log_name="predicted_label_eval",
                            cohorts=['cbbrain_validation', 'ab300'],
                            interval=250),
        ScheduledEvaluation(evaluator=SegmentationEvaluator("y_pred_eval", "y_eval"),
                            log_name="segmentation_eval",
                            cohorts=['cbbrain_validation', "ab300_validation", "fasd"],
                            interval=50),
        ScheduledEvaluation(evaluator=ContourImageEvaluator("Axial", "mean_dwi", "y_pred_eval", "y_eval",
                                                            slice_id=10, legend=True, ncol=3),
                            log_name="contour_image_01",
                            subjects=plot_subjects,
                            interval=10),
        ScheduledEvaluation(evaluator=ContourImageEvaluator("Coronal", "mean_dwi", "y_pred_eval", "y_eval",
                                                            slice_id=35, legend=True, ncol=1),
                            log_name="contour_image_02",
                            subjects=plot_subjects,
                            interval=10),
    ]

    def scoring_function(evaluation_dict):

        # Grab the output of the SegmentationEvaluator on the cbbrain and ab300 validation cohorts
        seg_eval_cbbrain = evaluation_dict['segmentation_eval']['cbbrain_validation']
        seg_eval_ab300 = evaluation_dict['segmentation_eval']['ab300_validation']

        # Get the mean dice for each label (the mean is across subjects)
        # The SegmentationEvaluator output includes a "summary_stats" dict with the following key structure
        # {summary_stat_name: {stat_name: {label_name: value}}}
        cbbrain_dice = seg_eval_cbbrain["summary_stats"]['mean']['dice']
        ab300_dice = seg_eval_ab300["summary_stats"]['mean']['dice']

        # Now take the mean across all labels
        from statistics import mean
        cbbrain_dice = mean(cbbrain_dice.values())
        ab300_dice = mean(ab300_dice.values())

        # Model must perform equally well on cbbrain and ab300
        score = (cbbrain_dice + ab300_dice) / 2
        return score

    train_predictor = StandardPredict(sagittal_split=True, image_names=['X', 'y'])
    validation_predictor = StandardPredict(sagittal_split=True, image_names=['X'])

    train_dataloader_factory = StandardDataLoader(sampler=RandomSampler)
    validation_dataloader_factory = StandardDataLoader(sampler=SequentialSampler)

    context.add_component("trainer", SegmentationTrainer,
                          training_batch_size=2,
                          save_rate=100,
                          scoring_interval=50,
                          scoring_function=scoring_function,
                          one_time_evaluators=one_time_evaluators,
                          training_evaluators=training_evaluators,
                          validation_evaluators=validation_evaluators,
                          max_iterations_with_no_improvement=500,                           
                          train_predictor=train_predictor,
                          validation_predictor=validation_predictor,
                          train_dataloader_factory=train_dataloader_factory,
                          validation_dataloader_factory=validation_dataloader_factory)

    return context
