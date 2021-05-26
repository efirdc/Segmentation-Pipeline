from context import Context
import torchio as tio
from datasets import SubjectFolder, ImageDefinition
from segmentation_training import SegmentationTrainer
from models import NestedResUNet
from evaluation import HybridLogisticDiceLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam


def get_context(device, variables, predict_hbt=False, **kwargs):
    context = Context(device, name="dmri-hippo", variables=variables, globals=globals())

    image_definitions = [
        ImageDefinition(name="mean_dwi", glob_pattern="mean_dwi.*"),
        ImageDefinition(name="md", glob_pattern="md.*"),
        ImageDefinition(name="fa", glob_pattern="fa.*"),
    ]
    label_definitions = [
        ImageDefinition(name="whole_roi", glob_pattern="whole_roi.*",
                        label_names={"left_whole": 1, "right_whole": 2}
                        ),
        ImageDefinition(name="hbt_roi", glob_pattern="hbt_roi.*",
                        label_names={"left_head": 1, "left_body": 2, "left_tail": 3, "right_head": 4, "right_body": 5,
                                     "right_tail": 6}
                        ),
    ]
    collate_images = {"X": ["mean_dwi", "md", "fa"]}
    collate_labels = {"y": "hbt_roi" if predict_hbt else "whole_roi"}

    transforms = tio.Compose([
        tio.RescaleIntensity((0, 1.), (0.5, 99.5)),
        # tio.RandomGamma(include=["mean_dwi"]),
        tio.RandomBiasField(coefficients=0.5, include=["mean_dwi"]),
        tio.RescaleIntensity((-1, 1), (0., 99.5)),
        # tio.Crop((14, 14, 24, 0, 0, 0)),
        tio.Crop((62, 62, 70, 58, 0, 0)),
        tio.RemapLabels({2: 1}, masking_method="Right", include="whole_roi"),
        tio.RemapLabels({4: 1, 5: 2, 6: 3}, masking_method="Right", include="hbt_roi"),
        # tio.RandomElasticDeformation(num_control_points=(7, 7, 5), image_interpolation="linear"),
        tio.Pad((0, 0, 0, 0, 2, 2), padding_mode="minimum"),
    ])

    val_transforms = tio.Compose([
        tio.RescaleIntensity((-1, 1.), (0.5, 99.5)),
        tio.Crop((62, 62, 70, 58, 0, 0)),
        tio.Pad((0, 0, 0, 0, 2, 2), padding_mode="minimum"),
    ])

    cbbrain_val_subjects = [f"cbbrain_{subject_id:03}" for subject_id in (
        32, 42, 55, 67, 82, 88, 96, 98, 102, 107, 110, 117, 123, 143, 145, 149, 173, 182, 184, 401
    )]
    dataset_params = dict(path="$DATASET_PATH", image_definitions=image_definitions,
                          label_definitions=label_definitions, collate_images=collate_images,
                          collate_labels=collate_labels, require_images=True)

    context.add_part("dataset", SubjectFolder, exclude_subjects=cbbrain_val_subjects,
                     transforms=transforms, include_attributes=dict(protocol="cbbrain"), **dataset_params)
    context.add_part("val_cbbrain", SubjectFolder, include_subjects=cbbrain_val_subjects,
                     transforms=val_transforms, **dataset_params)
    context.add_part("val_ab300", SubjectFolder, include_attributes=dict(protocol="ab300"),
                     transforms=val_transforms, **dataset_params)
    context.add_part("val_fasd", SubjectFolder, include_attributes=dict(pathologies="FASD"),
                     transforms=val_transforms, **dataset_params)

    context.add_part("datasampler", RandomSampler, data_source="self.dataset")
    context.add_part("dataloader", DataLoader, dataset="self.dataset", batch_size=4, sampler="self.datasampler",
                     drop_last=False, collate_fn="self.dataset.collate", pin_memory=False, num_workers=1,
                     persistent_workers=True)
    context.add_part("model", NestedResUNet, input_channels=3, output_channels=2, filters=40, dropout_p=0.2,
                     saggital_split=True)
    context.add_part("optimizer", Adam, params="self.model.parameters()", lr=0.0002)
    context.add_part("criterion", HybridLogisticDiceLoss)
    context.add_part("trainer", SegmentationTrainer,
                     save_folder="$CHECKPOINTS_PATH", sample_rate=50, save_rate=1000,
                     val_datasets=[
                         dict(dataset="self.val_cbbrain", log_prefix="Cb Val", preload=True, interval=50),
                         dict(dataset="self.val_ab300", log_prefix="Ab Val", preload=True, interval=50),
                         dict(dataset="self.val_fasd", log_prefix="FASD Val", preload=True, interval=250),
                     ],
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
