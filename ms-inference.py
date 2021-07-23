import argparse
from pathlib import Path

import torchio as tio
import torch

from models import EnsembleModels, EnsembleFlips, EnsembleOrientations
from post_processing import remove_holes, remove_small_components
from torch_context import TorchContext
from segmentation import patch_predict


def inference(dataset, model):

    for i in range(len(dataset)):
        subject = dataset[i]
        untransformed_subject = dataset.subjects[i]

        print(f"Running model for subject {subject['name']}")

        out_folder = args.out_folder
        if out_folder == "":
            out_folder = Path(subject["folder"])
        else:
            out_folder = Path(args.out_folder) / subject['name']
            out_folder.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            subject = patch_predict(model=model,
                                    device=device,
                                    subjects=[subject],
                                    patch_batch_size=1,
                                    patch_size=96,
                                    patch_overlap=48,
                                    padding_mode="edge",
                                    overlap_mode="average")[0]

        transform = subject.get_composed_history()
        inverse_transform = transform.inverse(warn=False)

        pred_subject = tio.Subject({'y': subject['y_pred']})
        inverse_pred_subject = inverse_transform(pred_subject)
        output_label = inverse_pred_subject.get_first_image()

        label_data = output_label['data'][0].numpy()

        label_data, hole_voxels_removed = remove_holes(label_data, hole_size=64)
        print(f"Filled {hole_voxels_removed} voxels from detected holes.")

        label_data, small_lesions_removed = remove_small_components(label_data, 3)
        print(f"Removed {small_lesions_removed} voxels from small predictions less than size 3.")

        label_data = torch.from_numpy(label_data[None]).to(torch.int32)
        output_label.set_data(label_data)

        target_image = untransformed_subject.get_first_image()
        output_label = tio.Resample(target_image)(output_label)

        if output_label.spatial_shape != target_image.spatial_shape:
            raise Warning(f"Segmentation shape and original image shape do not match.")

        print()

        output_label.save(out_folder / args.output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Hippocampus Segmentation")
    parser.add_argument("ensemble_path", type=str, help="Folder with models")
    parser.add_argument("dataset_path", type=str, help="Path to the subjects data folders.")
    parser.add_argument(
        "output_filename",
        type=str,
        help="File name for segmentation output. Can specify .nii or .nii.gz if compression is desired.",
    )
    parser.add_argument("--out_folder", type=str, default="", help="Folder for output.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device to use. Set to 'cpu' if there are issues with gpu usage. A specific gpu can be selected"
             " using 'cuda:0' or 'cuda:1' on a multi-gpu machine.",
    )
    parser.add_argument(
        "--ensemble_orientations",
        type=str,
        default="",
        help="Either 'flips' or 'orientations'"
    )
    parser.add_argument("--ensemble_folds", default=False, action='store_true')
    parser.add_argument("--cohort", type=str, default=None)
    args = parser.parse_args()
    print(args)

    if args.device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            device = torch.device("cpu")
            print("cuda not available, switched to cpu")
    else:
        device = torch.device(args.device)
    print("using device", device)

    ensemble_path = Path(args.ensemble_path)
    contexts = []
    for file_path in ensemble_path.iterdir():
        context = TorchContext(
            device, file_path=file_path, variables=dict(DATASET_PATH=args.dataset_path)
        )
        context.keep_components(('model', 'dataset'))
        context.init_components()

        if args.ensemble_orientations == 'orientations':
            context.model = EnsembleOrientations(context.model, strategy='majority')
        if args.ensemble_orientations == 'flips':
            context.model = EnsembleFlips(context.model, strategy='majority')

        contexts.append(context)
    print("Loaded models.")

    if args.ensemble_folds:
        context = contexts[0]
        models = [context.model for context in contexts]
        context.model = EnsembleModels(models, strategy='majority')
        context = [contexts]

    for i, context in enumerate(contexts):
        if args.cohort is None:
            dataset = context.dataset
        else:
            dataset = context.dataset.get_cohort_dataset(args.cohort)
        model = context.model
        model.eval()
        print(f"Running evaluation for context {i}")
        inference(dataset, model)
