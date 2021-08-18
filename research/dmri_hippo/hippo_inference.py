import argparse
import json
from pathlib import Path

import torch
import torchio as tio
import fire

from segmentation_pipeline.models.ensemble import EnsembleFlips, EnsembleModels
from segmentation_pipeline.post_processing import keep_components, remove_holes
from segmentation_pipeline.utils.torch_context import TorchContext


def inference(dataloader, predictor, model, device):

    model.eval()

    out_subjets = []
    for subjects in dataloader:

        with torch.no_grad():
            subjects, _ = predictor.predict(model=model, device=device, subjects=subjects)

        for subject in subjects:
            transform = subject.get_composed_history()
            inverse_transform = transform.inverse(warn=False)
            pred_subject = tio.Subject({"y": subject["y_pred"]})
            inverse_pred_subject = inverse_transform(pred_subject)
            output_label = inverse_pred_subject.get_first_image()
            subject["y_pred"].set_data(output_label["data"])
            out_subjets.append(subject)

    return out_subjets


def post_process(output_label):

    label_data = output_label["data"][0].numpy()

    label_data, hole_voxels_removed = remove_holes(label_data, hole_size=64)
    txt_output = f"Filled {hole_voxels_removed} voxels from detected holes.\n"

    num_components = label_data.max()
    label_data, num_components_removed, num_elements_removed = keep_components(label_data, num_components)
    txt_output += f"Removed {num_elements_removed} voxels from {num_components_removed} components."

    label_data = torch.from_numpy(label_data[None]).to(torch.int32)
    output_label.set_data(label_data)

    return txt_output


def generate_file_name(context, output_name):
    if output_name is None:
        file_name = context.name
    else:
        file_name = Path(output_name).stem

    return file_name


def save_subjects_predictions(subjects, out_folder, output_filename):

    for subject in subjects:

        if out_folder == "":
            out_folder = Path(subject["folder"])
        else:
            out_folder = Path(out_folder) / "subjects" / subject["name"]

        out_folder.mkdir(exist_ok=True, parents=True)

        subject["y_pred"].save(out_folder / (output_filename + ".nii.gz"))


def post_process_subjects(subjects, image_name):
    txt_output = ""
    for subject in subjects:
        txt_output += subject["name"] + "\n"
        txt_output += post_process(subject[image_name])

    return txt_output


def main(
    ensemble_path: str,
    dataset_path: str,
    output_filename: str = None,
    out_folder: str = "",
    device: str = "cpu",
    ensemble_flips: bool = False,
    ensemble_folds: bool = False,
    cohort: str = None,
    num_workers: int = 0,
    batch_size: int = 4,
):
    """Auto Hippocampus Segmentation  Run with `python -m research.dmri_hippo.hippo_inference` followed by args

    Args:
        ensemble_path: Folder with models
        dataset_path: Path to the subjects data folders.
        output_filename: File name for segmentation output. Provided extensions will be ignored and file will be saved ass .nii.gz.
            If output_filename is not provided the context name is used.
        out_folder: Folder for output.
        device: PyTorch device to use. Set to 'cpu' if there are issues with gpu usage. A specific gpu can be selected
            using 'cuda:0' or 'cuda:1' on a multi-gpu machine.
        ensemble_flips:
        ensemble_folds:
        cohort:
        num_workers: How many CPU threads to use for data loading, preprocessing, and augmentation.
        batch_size: How many subjects should be run through the model at once.
    """
    input_args = locals().copy()

    if device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device("cpu")
            print("cuda not available, switched to cpu")
    else:
        device = torch.device(device)
    print("using device", device)

    ensemble_path = Path(ensemble_path)
    contexts = []
    for file_path in ensemble_path.iterdir():
        context = TorchContext(device, file_path=file_path, variables=dict(DATASET_PATH=dataset_path))
        context.keep_components(("model", "trainer", "dataset"))
        context.init_components()
        context.model.eval()

        if ensemble_flips:
            context.model = EnsembleFlips(context.model, strategy="majority")

        contexts.append(context)
    print("Loaded models.")

    if ensemble_folds:
        context = contexts[0]
        models = [context.model for context in contexts]
        context.model = EnsembleModels(models, strategy="majority")
        contexts = [context]

    for i, context in enumerate(contexts):
        if cohort is None:
            dataset = context.dataset
        else:
            dataset = context.dataset.get_cohort_dataset(cohort)
        print(f"Running inference for context {i}")
        dataloader = context.trainer.validation_dataloader_factory.get_data_loader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )
        subjects = inference(dataloader, context.trainer.validation_predictor, context.model, device)

        txt_output = post_process_subjects(subjects, "y_pred")
        print(txt_output)
        base_file_name = generate_file_name(context, output_filename)

        with open(Path(out_folder) / (base_file_name + ".txt"), "w") as f:
            f.write(txt_output)

        save_subjects_predictions(subjects, out_folder, base_file_name)

        with open(Path(out_folder) / (base_file_name + ".json"), "w") as f:
            inference_settings = dict()
            inference_settings.update(input_args)
            inference_settings["context_name"] = context.name
            inference_settings["output_filename"] = base_file_name + ".nii.gz"
            json.dump(inference_settings, f, indent=4)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Auto Hippocampus Segmentation")
    # parser.add_argument("ensemble_path", type=str, help="Folder with models")
    # parser.add_argument("dataset_path", type=str, help="Path to the subjects data folders.")
    # parser.add_argument(
    #     "--output_filename",
    #     type=str,
    #     help="File name for segmentation output. Can specify .nii or .nii.gz if compression is desired."
    #     "If output_filename is not provided the context name is used.",
    # )
    # parser.add_argument("--out_folder", type=str, default="", help="Folder for output.")
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     default="cpu",
    #     help="PyTorch device to use. Set to 'cpu' if there are issues with gpu usage. A specific gpu can be selected"
    #     " using 'cuda:0' or 'cuda:1' on a multi-gpu machine.",
    # )
    # parser.add_argument("--ensemble_flips", default=False, action="store_true")
    # parser.add_argument("--ensemble_folds", default=False, action="store_true")
    # parser.add_argument("--cohort", type=str, default=None)
    # parser.add_argument(
    #     "--num_workers",
    #     type=int,
    #     default=0,
    #     help="How many CPU threads to use for data loading, preprocessing, and augmentation.",
    # )
    # parser.add_argument(
    #     "--batch_size", type=int, default=4, help="How many subjects should be run through the model at once."
    # )
    # args = parser.parse_args()
    # print(args)

    # main(**vars(args))

    
    fire.Fire(main)