from torch_context import TorchContext
import torchio as tio
from data_processing.subject_folder import SubjectFolder
from segmentation_trainer import SegmentationTrainer
from models import NestedResUNet
from evaluation import HybridLogisticDiceLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam

from transforms import *
from predictors import *
from dataLoaderFactory import *


def get_context(device, variables, **kwargs):
    context = TorchContext(device, name="qsm-dgm", variables=variables)

    image_definitions = [
        ImageDefinition(name="t1", glob_pattern="MPRAGE.*"),
        ImageDefinition(name="qsm", glob_pattern="QSM.*"),
    ]
    label_definitions = [
        ImageDefinition(name="dgm", glob_pattern="vB_PS_r.*",
                        label_values={'left_ventricle': 1, 'right_ventricle': 2, 'left_caudate': 3, 'right_caudate': 4,
                                     'left_putamen': 5, 'right_putamen': 6, 'left_thalamus': 7, 'right_thalamus': 8,
                                     'left_globus_pallidus': 9, 'right_globus_pallidus': 10, 'internal_capsule': 17,
                                     'left_red_nucleus': 19, 'right_red_nucleus': 20,
                                     'left_substantia_nigra': 21, 'right_substantia_nigra': 22,
                                     'left_dentate_nucleus': 23, 'right_dentate_nucleus': 24}
                        ),
        ImageDefinition(name="ic", glob_pattern="IC.*",
                        label_values={'internal_capsule': 17}
                        ),
        ImageDefinition(name="pulv", glob_pattern="pulv.*",
                        label_values={'left_thalamus_pulvinar': 7, 'right_thalamus_pulvinar': 8}
                        ),
    ]

    transforms = tio.Compose([
        tio.RescaleIntensity((-1, 1), (0.1, 99.9)),
        tio.Crop((68, 68, 72, 72, 16, 16)),
        CustomRemoveLabels(
            labels=['left_ventricle', 'right_ventricle', 'left_dentate_nucleus', 'right_dentate_nucleus'],
            include="dgm"
        ),
        MergeLabels(
            merge_labels=[('left_caudate', 'right_caudate'),
                          ('left_putamen', 'right_putamen'),
                          ('left_globus_pallidus', 'right_globus_pallidus'),
                          ('left_substantia_nigra', 'right_substantia_nigra')],
            right_masking_method='Right', include='dgm'
        ),
        CustomSequentialLabels(),
        ConcatenateImages(image_names=["t1", "qsm"], image_channels=[1, 1], new_image_name="X"),
        CopyProperty(image_name="dgm", new_image_name="y"),
        CustomOneHot(num_classes=10, include="y")
    ])
    val_transforms = transforms

    val_subjects = ["Cb_Brain_058", "Cb_Brain_106"]
    dataset_params = dict(path="$DATASET_PATH", image_definitions=image_definitions,
                          label_definitions=label_definitions, collate_attributes=["X", "y"], require_images=True)

    context.add_component("dataset", SubjectFolder, exclude_subjects=val_subjects, transforms=transforms,
                          **dataset_params)
    context.add_component("val_dataset", SubjectFolder, include_subjects=val_subjects, transforms=val_transforms,
                          **dataset_params)

    context.add_component("datasampler", RandomSampler, data_source="self.dataset")
    context.add_component("dataloader", DataLoader, dataset="self.dataset", batch_size=4, sampler="self.datasampler",
                          drop_last=False, collate_fn="self.dataset.collate", pin_memory=False, num_workers=0,
                          persistent_workers=False)
    context.add_component("model", NestedResUNet, input_channels=2, output_channels=10, filters=40, dropout_p=0.2,
                          saggital_split=False)
    context.add_component("optimizer", Adam, params="self.model.parameters()", lr=0.0002)
    context.add_component("criterion", HybridLogisticDiceLoss)

    train_predictor = StandardPredict(device, image_names=['X', 'y'])
    val_predictor = StandardPredict(device, image_names=['X'])

    train_dataloader_factory = StandardDataLoader(sampler=RandomSampler, collate_fn=dont_collate)
    val_dataloader_factory = StandardDataLoader(sampler=RandomSampler, collate_fn=dont_collate)

    context.add_component("trainer", SegmentationTrainer, save_folder="$CHECKPOINTS_PATH", sample_rate=50, save_rate=250,
                          val_datasets=[
                         dict(dataset="self.val_dataset", log_prefix="Val", preload=True, interval=50),
                     ],
                          val_images=[
                         dict(interval=50, log_name="image0",
                              plane="Axial", image_name="qsm", slice=9, legend=True, ncol=1,
                              subjects=val_subjects),
                         dict(interval=50, log_name="image1",
                              plane="Coronal", image_name="qsm", slice=51, legend=True, ncol=1,
                              subjects=val_subjects),
                     ], 
                          train_predictor=train_predictor,
                          val_predictor=val_predictor,
                          train_dataloader_factory=train_dataloader_factory,
                          val_dataloader_factory=val_dataloader_factory)

    return context
