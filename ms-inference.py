import argparse
from pathlib import Path
import shutil
import subprocess

CONTEXT_PATH = Path("/Segmentation-Pipeline/saved_models/ms-seg.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect new MS lesions from two FLAIR images.")

    parser.add_argument("-t1", "--time01", type=str, help="First time step (path to the FLAIR image).", required=True)
    parser.add_argument("-t2", "--time02", type=str, help="Second time step (path to the FLAIR image).", required=True)
    parser.add_argument("-o", "--output", type=str, help="Path of the output segmentation.", required=True)
    parser.add_argument("-d", "--data_folder", type=str, help="Path of the data folder.", default="data/")

    args = parser.parse_args()

    flair_time01 = Path(args.time01)
    flair_time02 = Path(args.time02)
    output = Path(args.output)
    data_folder = Path(args.data_folder)

    input_folder = data_folder / "input" / "raw_data"

    subject_folder = input_folder / "01"
    subject_folder.mkdir(exist_ok=True, parents=True)

    input_file01 = subject_folder / "flair_time01_on_middle_space.nii.gz"
    shutil.copy(flair_time01, input_file01)

    input_file02 = subject_folder / "flair_time02_on_middle_space.nii.gz"
    shutil.copy(flair_time02, input_file02)

    output_folder = data_folder / "output"
    output_folder.mkdir(exist_ok=True)

    processed_folder = data_folder / "input" / "processed"
    processed_folder.mkdir(exist_ok=True, parents=True)

    p1 = subprocess.run(
        [
            "python", "/Segmentation-Pipeline/Anima-Scripts-Public/ms_lesion_segmentation/animaMSLongitudinalPreprocessing.py",
            "-i", input_folder,
            "-o", processed_folder,
        ]
    )

    p2 = subprocess.run(
        [
            "python", "/Segmentation-Pipeline/inference.py",
            CONTEXT_PATH,
            processed_folder,
            "temp.nii",
            "--out_folder", output_folder,
            "--keep_isolated_components"
        ]
    )

    outputFile = output_folder / "01" / "temp.nii"

    shutil.copy(outputFile, output)
