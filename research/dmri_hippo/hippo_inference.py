import argparse
import json
import pathlib
from pathlib import Path

import torch
import torchio as tio
import fire

from segmentation_pipeline import *

# Hack so that models can load on windows
pathlib.PosixPath = pathlib.Path


def inference(subjects, predictor, model, device):
    model.eval()

    subject_names = [subject['name'] for subject in subjects]
    print(f"running inference for subjects: {subject_names}")

    with torch.no_grad():
        subjects, _ = predictor.predict(model=model, device=device, subjects=subjects)

    for subject in subjects:
        transform = subject.get_composed_history()
        inverse_transform = transform.inverse(warn=False)
        pred_subject = tio.Subject({"y": subject["y_pred"]})
        inverse_pred_subject = inverse_transform(pred_subject)
        output_label = inverse_pred_subject.get_first_image()
        subject["y_pred"].set_data(output_label["data"].to(torch.int32))

    return subjects


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
            out_folder_path = Path(subject["folder"])
        else:
            out_folder_path = Path(out_folder) / "subjects" / subject["name"]

        out_folder_path.mkdir(exist_ok=True, parents=True)

        subject["y_pred"].save(out_folder_path / (output_filename + ".nii.gz"))


def post_process_subjects(subjects, image_name):
    txt_output = ""
    for subject in subjects:
        txt_output += subject["name"] + "\n"
        txt_output += post_process(subject[image_name]) + "\n"

    return txt_output


def main(
    ensemble_path: str,
    dataset_path: str,
    run_name: str,
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
        run_name:
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
    input_args['cohort'] = str(input_args['cohort'])

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

        if ensemble_flips:
            context.model = EnsembleFlips(context.model, strategy="majority", spatial_dims=(3, 4))

        contexts.append(context)
    print("Loaded models.")

    if ensemble_folds:
        context = contexts[0]
        models = [context.model for context in contexts]
        context.model = EnsembleModels(models, strategy="majority")
        names = [context.name for context in contexts]
        context.name = names
        contexts = [context]

    for i, context in enumerate(contexts):
        if cohort is None:
            dataset = context.dataset
        else:
            dataset = context.dataset.get_cohort_dataset(cohort)
        print(f"Running inference for context {context.name}")

        dataloader = context.trainer.validation_dataloader_factory.get_data_loader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

        for subjects in dataloader:
            subjects = inference(subjects, context.trainer.validation_predictor, context.model, device)

            base_file_name = generate_file_name(context, output_filename)

            save_subjects_predictions(subjects, out_folder, base_file_name + "_before_processing")

            txt_output = post_process_subjects(subjects, "y_pred")
            print(txt_output)

            if output_filename is None:
                mode = "w"
            else:
                mode = "a"
            with open(Path(out_folder) / (base_file_name + ".txt"), mode) as f:
                f.write(txt_output)

            save_subjects_predictions(subjects, out_folder, base_file_name)

    base_file_name = generate_file_name(context, output_filename)
    with open(Path(out_folder) / (run_name + ".json"), "w") as f:
        inference_settings = dict()
        inference_settings.update(input_args)
        inference_settings["context_name"] = [context.name for context in contexts]
        inference_settings["output_filename"] = base_file_name + ".nii.gz"
        json.dump(inference_settings, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
