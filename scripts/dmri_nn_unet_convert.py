import argparse

import torch

from segmentation_pipeline import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dmri hippo splits.")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--skip_saving_images", default=False, action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    config = load_module("../configs/diffusion_hippocampus.py")

    variables = dict(DATASET_PATH=args.dataset_path)
    context = config.get_context(device, variables)
    context.init_components()

    save_dataset_as_nn_unet(
        context.dataset,
        args.output_path,
        short_name="DMRI",
        image_names=['mean_dwi', 'md', 'fa'],
        label_map_name='whole_roi',
        cross_validation_cohort='cross_validation',
        test_cohort='cbbrain_test',
        output_folds=True,
        num_folds=5,
        skip_saving_images=args.skip_saving_images
    )
