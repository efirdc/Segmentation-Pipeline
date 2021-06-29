import torchio as tio
from torch.optim import Adam
import torch

from context import Context
from segmentation_training import SegmentationTrainer, ScheduledEvaluation
from models import NestedResUNet
from evaluation import HybridLogisticDiceLoss
from transforms import *
from data_processing import *
from evaluators import *


def get_context(device, variables, **kwargs):
    context = Context(device, name="msseg2", variables=variables, globals=globals())

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

    # Selected with random.shuffle
    validation_subjects = ['021', '035', '032', '068', '100', '051', '094', '030']

    cohorts = {}
    cohorts['all'] = RequireAttributes(input_images + output_labels)
    cohorts['training'] = ComposeFilters([
        RequireAttributes(output_labels),
        ForbidAttributes({"name": validation_subjects})
    ])
    cohorts['validation'] = RequireAttributes({"name": validation_subjects})

    common_transforms = tio.Compose([
        SetDataType(torch.float),
        CropToMask('brain_mask'),
        tio.Resample(1),
        #tio.EnsureShapeMultiple(target_multiple=8, method='crop'),
        tio.CropOrPad((128, 160, 128)),
        tio.RescaleIntensity((-1, 1.), (0.0, 99.5)),
        ConcatenateImages(image_names=["flair_time01", "flair_time02"], image_channels=[1, 1], new_image_name="X"),
        RenameProperty(old_name='ground_truth', new_name='y'),
        CustomOneHot(include="y")
    ])

    transforms = {
        'default': tio.Compose([
            common_transforms
        ]),
        'training': tio.Compose([
            common_transforms
        ]),
    }

    context.add_part("dataset", SubjectFolder, root='$DATASET_PATH', subject_path="",
                     subject_loader=subject_loader, cohorts=cohorts, transforms=transforms)
    context.add_part("model", NestedResUNet, input_channels=2, output_channels=2,
                     filters=40, dropout_p=0.2, saggital_split=False)
    context.add_part("optimizer", Adam, params="self.model.parameters()", lr=0.001)
    context.add_part("criterion", HybridLogisticDiceLoss)

    training_evaluators = [
        ScheduledEvaluation(evaluator=SegmentationEvaluator('y_pred_eval', 'y_eval'),
                            log_name='training_segmentation_eval'),
        ScheduledEvaluation(evaluator=LabelMapEvaluator('y_pred_eval'),
                            log_name='training_label_eval'),
    ]

    validation_evaluators = [
        ScheduledEvaluation(evaluator=SegmentationEvaluator("y_pred_eval", "y_eval"),
                            log_name="segmentation_eval",
                            cohorts=["validation"],
                            interval=25),
    ]

    contour_eval_params = [('021', "Coronal", 104), ('035', "Axial", 46), ('032', "Axial", 83),
                           ('068', "Saggital", 70), ('100', "Axial", 45), ('051', "Axial", 61),
                           ('094', "Coronal", 110), ('030', "Axial", 86)]
    validation_evaluators += [
        ScheduledEvaluation(evaluator=ContourImageEvaluator(plane, 'flair_time02', 'y_pred_eval', 'y_eval',
                                                            slice_id, legend=True, ncol=1),
                            log_name=f"contour_image_{subject_name}",
                            subjects=[subject_name],
                            interval=10)
        for subject_name, plane, slice_id in contour_eval_params
    ]

    def scoring_function(evaluation_dict):
        # Grab the output of the SegmentationEvaluator
        seg_eval = evaluation_dict['segmentation_eval']['validation']

        # Get the mean dice for each label (the mean is across subjects)
        # The SegmentationEvaluator output includes a "summary_stats" dict with the following key structure
        # {summary_stat_name: {stat_name: {label_name: value}}}
        dice = seg_eval["summary_stats"]['mean']['dice']

        # Now take the mean across all labels
        from statistics import mean
        dice = mean(dice.values())

        score = dice
        return score

    context.add_part("trainer", SegmentationTrainer, training_batch_size=1,
                     save_folder="$CHECKPOINTS_PATH", save_rate=100, scoring_interval=50,
                     scoring_function=scoring_function, one_time_evaluators=[],
                     training_evaluators=training_evaluators, validation_evaluators=validation_evaluators,
                     max_iterations_with_no_improvement=500)

    return context
