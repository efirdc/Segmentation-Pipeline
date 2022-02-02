import os

import torchio as tio
from torch.optim import SGD

from . import main_config as base_config
from segmentation_pipeline import *


def get_context(
        device,
        variables,
        prior_label_name,
        model_type=None,
        **kwargs
):
    context = base_config.get_context(device, variables, **kwargs)
    context.file_paths.append(os.path.abspath(__file__))
    context.config.update({
        'prior_label_name': prior_label_name,
        'model_type': model_type,
        "optimizer": "SGD"
    })

    # training_transform is a tio.Compose where the second transform is the augmentation
    dataset_defn = context.get_component_definition("dataset")

    subject_loader = dataset_defn['params']['subject_loader']
    subject_loader.loaders.append(
        ImageLoader(glob_pattern=f"$PREDICTIONS_PATH/subjects/$SUBJECT_NAME/{prior_label_name}.*",
                    image_name="y_prior",
                    image_constructor=tio.LabelMap,
                    label_values={"left_whole": 1, "right_whole": 2})
    )

    default_transform = dataset_defn['params']['transforms']['default']
    common_transforms_1, common_transforms_2 = default_transform.transforms

    # common_transforms_1.transforms[0] = tio.CropOrPad((96, 64, 24), padding_mode="minimum", mask_name='y_prior')
    common_transforms_1.transforms[1].include += ['y_prior']
    common_transforms_2.transforms += [
        CustomOneHot(include=['y_prior'])
    ]

    output_channels = 4 if kwargs['predict_hbt'] else 2
    model_defn = context.get_component_definition("model")
    if model_type is None:
        model_params = model_defn['params']
        model_params['output_channels'] = output_channels * output_channels
        model_params['hypothesis_class'] = StochasticMatrix
        model_params['hypothesis_params'] = {"channels": output_channels}
    elif model_type == "basic_unet":
        model_defn['constructor'] = ModularUNet
        model_defn['params'] = {
            'in_channels': 3,
            'out_channels': output_channels * output_channels,
            'filters': [40, 80, 120],
            'depth': 3,
            'block_params': {'residual': True},
            'downsample_class': BlurConv3d,
            'downsample_params': {'kernel_size': 3, 'stride': 2, 'padding': 1},
            'upsample_class': BlurConvTranspose3d,
            'upsample_params': {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 0},
            'hypothesis_class': StochasticMatrix,
            'hypothesis_params': {"channels": output_channels, "diag_bias": 5}
        }
    else:
        raise ValueError()

    optimizer_defn = context.get_component_definition("optimizer")
    optimizer_defn['constructor'] = SGD
    optimizer_defn['params'] = {'params': "self.model.parameters()", "lr": 0.01, "momentum": 0.95}

    trainer_defn = context.get_component_definition("trainer")
    trainer_params = trainer_defn['params']
    trainer_params['train_predictor'] = StandardPredict(sagittal_split=True, image_names=['X', 'y'],
                                                        refine_image="y_prior")
    trainer_params['validation_predictor'] = StandardPredict(sagittal_split=True, image_names=['X'],
                                                             refine_image="y_prior")

    return context
