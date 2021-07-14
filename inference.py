import argparse
import itertools
import os
from pathlib import Path

import torch
import torchio as tio
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from data_processing import *
from models.ensemble import Ensemble
from post_processing import *
from torch_context import TorchContext
from transforms import *
from utils import collate_subjects, dont_collate, filter_transform


def segmentation_predict(context, input_data):

    with torch.no_grad():
        probs = context.model(input_data.to(context.device))[0]

    return probs.cpu()


def prepare_output_folder(subject, out_folder):
    if out_folder == "":
        out_folder = Path(subject["folder"])
    else:
        out_folder = Path(args.out_folder)
        out_folder = out_folder / subject["name"]
        out_folder.mkdir(exist_ok=True, parents=True)

    return out_folder


def get_test_time_transform():
    for dim_perm in itertools.permutations((0, 1, 2)):
        for flip_mask in itertools.product([False, True], repeat=3):
            flip_axis = np.array([0, 1, 2])
            axes_to_flip = flip_axis[np.array(flip_mask)]

            transforms = [PermuteDimensions(dim_perm)]

            if axes_to_flip.size != 0:
                transforms.append(tio.Flip(tuple(axes_to_flip.tolist())))

            composeTrans = tio.Compose(transforms)

            yield composeTrans


def getEnsembleContext(ensemble_path):
    for file in ensemble_path.iterdir():

        context = TorchContext(
            device, file_path=file, variables=dict(DATASET_PATH=args.dataset_path, CHECKPOINTS_PATH="")
        )
        context.init_components()
        models.append(context.model)

    ensemble_model = Ensemble(models)

    context.model = ensemble_model
    return context


def test_time_augmentation(context, subject):
    results = []

    for transform in get_test_time_transform():

        augmented = transform(subject)

        if context.trainer.enable_patch_mode:
            probs = context.trainer.patch_predict(context, augmented, 8, 48)
        else:
            batch = collate_subjects(augmented, image_names=["X"], device=context.device)
            probs = segmentation_predict(context, batch["X"])

        output_tensor = probs.argmax(dim=0, keepdim=True).cpu()
        lm_temp = tio.LabelMap(tensor=torch.rand(1, 1, 1, 1))
        augmented.add_image(lm_temp, "label")
        augmented.label.set_data(output_tensor)
        back = augmented.apply_inverse_transform(warn=False)
        results.append(back.label.data)

    result = torch.stack(results).long()
    out = result.mode(dim=0).values
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Hippocampus Segmentation")
    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument("dataset_path", type=str, help="Path to the subjects data folders.")
    parser.add_argument(
        "output_filename",
        type=str,
        help="File name for segmentation output. " "Can specify .nii or .nii.gz if compression is desired.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device to use. Set to 'cpu' if there are issues with gpu usage. A specific gpu can be selected"
        " using 'cuda:0' or 'cuda:1' on a multi-gpu machine.",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="",
        help="Redirect all output to a folder. Otherwise, the output will be placed in each subjects folder.",
    )
    parser.add_argument(
        "--keep_isolated_components",
        dest="remove_isolated_components",
        action="store_false",
        help="Don't remove isolated components in the post processing pipeline. (on by default)",
    )
    parser.set_defaults(remove_isolated_components=True)
    parser.add_argument(
        "--keep_holes",
        dest="remove_holes",
        action="store_false",
        help="Don't remove holes in the post processing pipeline. (on by default)",
    )
    parser.set_defaults(remove_holes=True)
    parser.add_argument(
        "--output_raw_probabilities",
        dest="output_probabilities",
        action="store_true",
        help="Output the raw probabilities from the network instead of converting them to a segmentation map",
    )
    parser.set_defaults(output_probabilities=False)
    parser.add_argument(
        "--disable-test-time-augmentation",
        dest="test_time_augmentation",
        action="store_false",
        help="Disable test time augmentation where a prediction is the combination of multiple augmentations",
    )
    parser.set_defaults(test_time_augmentation=True)
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            device = torch.device("cpu")
            print("cuda not available, switched to cpu")
    else:
        device = torch.device(args.device)
    print("using device", device)

    model_path = Path(args.model_path)
    models = []
    if model_path.is_dir():
        context = getEnsembleContext(model_path)
    else:

        context = TorchContext(
            device, file_path=model_path, variables=dict(DATASET_PATH=args.dataset_path, CHECKPOINTS_PATH="")
        )
        context.init_components()

    dataset = context.dataset
    dataset.transforms["default"] = filter_transform(dataset.transforms["default"], exclude_types=[TargetResample])
    dataset.set_transform("default")

    test_dataloader = DataLoader(
        dataset=dataset, batch_size=1, sampler=SequentialSampler(dataset), collate_fn=dont_collate
    )

    total = len(context.dataset)
    pbar = tqdm(total=total)
    context.model.eval()
    for i, subjects in enumerate(test_dataloader):

        assert len(subjects) == 1, "Batch size must be 1"

        subject = subjects[0]
        pbar.write(f"subject {subject['name']}: ")
        out_folder = prepare_output_folder(subject, args.out_folder)

        if args.test_time_augmentation:
            out = test_time_augmentation(context, subject)
            out = out.squeeze().numpy()

        else:
            raise NotImplementedError
            if context.trainer.enable_patch_mode:
                probs = context.trainer.patch_predict(context, subject, 8, 12)
            else:
                batch = collate_subjects(subjects, image_names=["X"], device=context.device)
                probs = segmentation_predict(context, batch["X"])

            out = torch.argmax(probs, dim=0)
            out = out.numpy()
            out = torch.from_numpy(out).unsqueeze(0)
            out = out.int()
            image = tio.LabelMap(tensor=out)
            inverse_transforms = subject.get_composed_history().inverse(warn=False)
            image = inverse_transforms(image)

        if not args.output_probabilities:

            if args.remove_holes:
                out, hole_voxels_removed = remove_holes(out, hole_size=64, return_counts=True)
                pbar.write(f"\tFilled {hole_voxels_removed} voxels from detected holes.")

            out = torch.from_numpy(out).unsqueeze(0)
            out = out.int()
            image = tio.LabelMap(tensor=out)

        else:
            raise NotImplementedError
            image = tio.ScalarImage(tensor=probs)

        # inverse_transforms = subject.get_composed_history().inverse(warn=False)
        # image = inverse_transforms(image)

        orig_subject = dataset.subjects_map[subject["name"]]  # access subject without applying transformations

        # compare shapes ignoring the channel dimension
        if image.shape[1:] != orig_subject.shape[1:]:
            resample_transform = tio.Resample(orig_subject.get_images()[0])
            image = resample_transform(image)

        assert orig_subject.shape[1:] == image.shape[1:], "Segmentation shape and original image shape do not match"

        pbar.write("\tSaving image...")
        image.save(out_folder / args.output_filename)
        pbar.write("\tFinished subject")
        pbar.update(1)
