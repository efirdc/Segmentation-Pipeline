import argparse
import os
from pathlib import Path

import torch
import torchio as tio
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from data_processing import *
from post_processing import *
from torch_context import TorchContext
from segmentation import collate_subjects, dont_collate


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

    context = TorchContext(
        device, file_path=args.model_path, variables=dict(DATASET_PATH=args.dataset_path, CHECKPOINTS_PATH="")
    )

    context.init_components()
    dataset = context.dataset

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

        if context.trainer.enable_patch_mode:
            probs = context.trainer.patch_predict(context, subject, 32)
        else:
            batch = collate_subjects(subjects, image_names=["X"], device=context.device)
            probs = segmentation_predict(context, batch["X"])

        if not args.output_probabilities:

            out = torch.argmax(probs, dim=0)
            out = out.numpy()

            if args.remove_isolated_components:
                num_components = out.max()
                out, components_removed, component_voxels_removed = keep_components(
                    out, num_components
                )
                pbar.write(
                    f"\tRemoved {component_voxels_removed} voxels from "
                    f"{components_removed} detected isolated components."
                )

            if args.remove_holes:
                out, hole_voxels_removed = remove_holes(out, hole_size=64)
                pbar.write(f"\tFilled {hole_voxels_removed} voxels from detected holes.")

            out = torch.from_numpy(out).unsqueeze(0)
            out = out.int()
            image = tio.LabelMap(tensor=out)

        else:
            image = tio.ScalarImage(tensor=probs)

        inverse_transforms = subject.get_composed_history().inverse(warn=False)
        image = inverse_transforms(image)

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
