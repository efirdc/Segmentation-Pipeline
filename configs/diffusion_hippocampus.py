from context import Context
import torchio as tio
from segmentation_training import SegmentationTrainer
from models import NestedResUNet
from evaluation import HybridLogisticDiceLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam

from transforms import *
from data_processing import *


def get_context(device, variables, predict_hbt=False, **kwargs):
    context = Context(device, name="dmri-hippo", variables=variables, globals=globals())

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

    cbbrain_validation_subjects = [f"cbbrain_{subject_id:03}" for subject_id in (
        32, 42, 55, 67, 82, 88, 96, 98, 102, 107, 110, 117, 123, 143, 145, 149, 173, 182, 184, 401
    )]

    training_subject_filter = ComposeFilters([
        RequireAttributes(['mean_dwi', 'md', 'fa', 'whole_roi', 'hbt_roi']),
        RequireAttributes({"pathologies": "None", "protocol": "cbbrain", "rescan_id": "None"}),
        ForbidAttributes({"name": cbbrain_validation_subjects})
    ])
    validation_subject_filter = NegateFilter(training_subject_filter)

    common_transforms = tio.Compose([
        tio.Crop((62, 62, 70, 58, 0, 0)),
        tio.Pad((0, 0, 0, 0, 2, 2), padding_mode="minimum"),
        MergeLabels([('left_whole', 'right_whole')], right_masking_method="Right", include="whole_roi"),
        MergeLabels([('left_head', 'right_head'), ('left_body', 'right_body'), ('left_tail', 'right_tail')],
                    right_masking_method="Right", include="hbt_roi"),
        ConcatenateImages(image_names=["mean_dwi", "md", "fa"], image_channels=[1, 1, 1], new_image_name="X"),
        CopyImage(image_name="hbt_roi" if predict_hbt else "whole_roi", new_image_name="y"),
        CustomOneHot(num_classes=4 if predict_hbt else 2, include="y")
    ])

    training_transforms = tio.Compose([
        tio.RescaleIntensity((0, 1.), (0.5, 99.5)),
        tio.RandomBiasField(coefficients=0.5, include=["mean_dwi"]),
        tio.RescaleIntensity((-1, 1), (0., 99.5)),
        common_transforms
    ])

    val_transforms = tio.Compose([
        tio.RescaleIntensity((-1, 1.), (0.5, 99.5)),
        common_transforms
    ])

    dataset_params = dict(path="$DATASET_PATH", subject_path="subjects",
                          subject_loader=subject_loader, collate_attributes=["X", "y"])

    context.add_part("train_dataset", SubjectFolder, subject_filter=training_subject_filter,
                     transforms=training_transforms, **dataset_params)
    context.add_part("val_dataset", SubjectFolder, subject_filter=validation_subject_filter,
                     transforms=val_transforms, **dataset_params)

    context.add_part("datasampler", RandomSampler, data_source="self.dataset")
    context.add_part("dataloader", DataLoader, dataset="self.dataset", batch_size=4, sampler="self.datasampler",
                     drop_last=False, collate_fn="self.dataset.collate", pin_memory=False, num_workers=0,
                     persistent_workers=False)
    context.add_part("model", NestedResUNet, input_channels=3, output_channels=4 if predict_hbt else 2,
                     filters=40, dropout_p=0.2, saggital_split=True)
    context.add_part("optimizer", Adam, params="self.model.parameters()", lr=0.0002)
    context.add_part("criterion", HybridLogisticDiceLoss)

    cbbrain_validation_filter = RequireAttributes({'name': cbbrain_validation_subjects})
    ab300_supervised_filter = ComposeFilters([
        RequireAttributes(['mean_dwi', 'md', 'fa', "whole_roi", "hbt_roi"]),
        RequireAttributes({"protocol": "ab300"})
    ])
    ab300_unsupervised_filter = ComposeFilters([
        RequireAttributes(['mean_dwi', 'md', 'fa']),
        ForbidAttributes(["whole_roi", "hbt_roi"]),
        RequireAttributes({"protocol": "ab300"})
    ])
    fasd_filter = ComposeFilters(
        RequireAttributes(['mean_dwi', 'md', 'fa', "whole_roi", "hbt_roi"]),
        RequireAttributes({"pathologies": "fasd"})
    )
    plot_subjects = ["cbbrain_042", "cbbrain_082", "cbbrain_143",
                     "cbbrain_036", "cbbrain_039", "cbbrain_190",  # FASD
                     "ab300_002", "ab300_005", "ab300_090"]

    context.add_part("trainer", SegmentationTrainer,
                     save_folder="$CHECKPOINTS_PATH", sample_rate=50, save_rate=1000,
                     training_evaluators=[SegmentationEvaluator],
                     val_images=[
                         dict(interval=50, log_name="image0",
                              plane="Axial", image_name="mean_dwi", slice=10, legend=True, ncol=3,
                              subjects=["cbbrain_042", "cbbrain_082", "cbbrain_143",
                                        "cbbrain_036", "cbbrain_039", "cbbrain_190",  # FASD
                                        "ab300_002", "ab300_005", "ab300_090"]),
                         dict(interval=50, log_name="image1",
                              plane="Coronal", image_name="mean_dwi", slice=35, legend=True, ncol=1,
                              subjects=["cbbrain_042", "cbbrain_082", "cbbrain_143",
                                        "cbbrain_036", "cbbrain_039", "cbbrain_190",  # FASD
                                        "ab300_002", "ab300_005", "ab300_090"]),
                     ])

    return context
