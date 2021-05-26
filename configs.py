from context import Context
import torchio as tio
from datasets import SubjectFolder, ImageDefinition
from segmentation_training import SegmentationTrainer
from models import NestedResUNet
from evaluation import HybridLogisticDiceLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam


def get_context(config_name, device, variables, **kwargs):
    context = Context(device, name=config_name, variables=variables, globals=globals())

    if config_name == "dmri-hippo":
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
                              label_definitions=label_definitions, require_images=True)

        context.add_part("dataset", SubjectFolder, exclude_subjects=cbbrain_val_subjects,
                         transforms=transforms, include_attributes=dict(protocol="cbbrain"), **dataset_params)
        context.add_part("val_cbbrain", SubjectFolder, include_subjects=cbbrain_val_subjects,
                         transforms=val_transforms, **dataset_params)
        context.add_part("val_ab300", SubjectFolder, include_attributes=dict(protocol="ab300"),
                         transforms=val_transforms, **dataset_params)
        context.add_part("val_fasd", SubjectFolder, include_attributes=dict(pathologies="FASD"),
                         transforms=val_transforms, **dataset_params)

        context.add_part("datasampler", RandomSampler, data_source=ct("dataset"))
        context.add_part("dataloader", DataLoader, dataset=ct("dataset"), batch_size=4, sampler=ct("datasampler"),
                         drop_last=False, collate_fn=None, pin_memory=False, num_workers=10,
                         persistent_workers=True)
        context.add_part("model", NestedResUNet, input_channels=3, output_channels=2, filters=40, dropout_p=0.2,
                         saggital_split=True)
        context.add_part("optimizer", Adam, params=ct("model.parameters()"), lr=0.0002)
        context.add_part("criterion", HybridLogisticDiceLoss)
        context.add_part("trainer", SegmentationTrainer,
                         input_images=["mean_dwi", "md", "fa"], target_label="whole_roi",
                         save_folder="$CHECKPOINTS_PATH", sample_rate=50, save_rate=1000,
                         val_datasets=[
                             dict(dataset=ct("val_cbbrain"), log_prefix="Cb Val", preload=True, interval=50),
                             dict(dataset=ct("val_ab300"), log_prefix="Ab Val", preload=True, interval=50),
                             dict(dataset=ct("val_fasd"), log_prefix="FASD Val", preload=True, interval=250),
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

    elif config_name == 'dgm':
        image_definitions = [
            ImageDefinition(name="t1", glob_pattern="MPRAGE.*"),
            ImageDefinition(name="qsm", glob_pattern="QSM.*"),
        ]
        label_definitions = [
            ImageDefinition(name="dgm", glob_pattern="vB_PS_r.*",
                 label_names={'left_ventricle': 1, 'right_ventricle': 2, 'left_caudate': 3, 'right_caudate': 4,
                              'left_putamen': 5, 'right_putamen': 6,
                              'left_thalamus': 7, 'right_thalamus': 8, 'left_globus_pallidus': 9,
                              'right_globus_pallidus': 10, 'internal_capsule': 17,
                              'left_red_nucleus': 19, 'right_red_nucleus': 20, 'left_substantia_nigra': 21,
                              'right_substantia_nigra': 22,
                              'left_dentate_nucleus': 23, 'right_dentate_nucleus': 24}
                 ),
            ImageDefinition(name="ic", glob_pattern="IC.*",
                 label_names={'internal_capsule': 17}
                 ),
            ImageDefinition(name="pulv", glob_pattern="pulv.*",
                 label_names={'left_thalamus_pulvinar': 7, 'right_thalamus_pulvinar': 8}
                 ),
        ]

        transforms = tio.Compose([
            tio.RescaleIntensity((-1, 1), (0.1, 99.9)),
            tio.Crop((68, 68, 72, 72, 16, 16)),
            tio.RemoveLabels([1, 2, 23, 24], include="dgm"),
            tio.RemapLabels({4: 3, 6: 5, 10: 9, 22: 21}, masking_method="Right", include="dgm"),
            tio.SequentialLabels(),
        ])
        val_transforms = transforms

        val_subjects = ["Cb_Brain_058", "Cb_Brain_106"]
        dataset_params = dict(path="$DATASET_PATH", image_definitions=image_definitions,
                              label_definitions=label_definitions,
                              require_images=True, input_images=["t1", "qsm"], target_label="dgm")

        context.add_part("dataset", SubjectFolder, exclude_subjects=val_subjects, transforms=transforms,
                         **dataset_params)
        context.add_part("val_dataset", SubjectFolder, include_subjects=val_subjects, transforms=val_transforms,
                         **dataset_params)
        context.add_part("datasampler", RandomSampler, data_source=ct("dataset"))
        context.add_part("dataloader", DataLoader, dataset=ct("dataset"), batch_size=4, sampler=ct("datasampler"),
                         drop_last=False, collate_fn=SubjectFolder.collate, pin_memory=False, num_workers=10,
                         persistent_workers=True)
        context.add_part("model", NestedResUNet, input_channels=2, output_channels=10, filters=40, dropout_p=0.2,
                         saggital_split=False)
        context.add_part("optimizer", Adam, params=ct("model.parameters()"), lr=0.0002)
        context.add_part("criterion", HybridLogisticDiceLoss)
        context.add_part("trainer", SegmentationTrainer, save_folder="$CHECKPOINTS_PATH", sample_rate=50, save_rate=250,
                         preload_training_dataset=True,
                         val_datasets=[
                             dict(dataset=ct("val_dataset"), log_prefix="Val", preload=True, interval=50),
                         ],
                         val_images=[
                             dict(interval=50, log_name="image0",
                                  plane="Axial", image_name="qsm", slice=9, legend=True, ncol=1,
                                  subjects=val_subjects),
                             dict(interval=50, log_name="image1",
                                  plane="Coronal", image_name="qsm", slice=51, legend=True, ncol=1,
                                  subjects=val_subjects),
                         ])
    else:
        raise ValueError(f"Configuration for {config_name} does not exist.")

    return context
