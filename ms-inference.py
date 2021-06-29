import argparse
from pathlib import Path
import shutil
from subprocess import call

import torchio as tio
import torch

CONTEXT_PATH = "saved_models/iter00000000.pt"

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

    input_folder = data_folder / "input" / "raw_data" / "01"
    input_folder.mkdir(exist_ok=True, parents=True)

    input_file01 = input_folder / flair_time01.name
    shutil.copy(flair_time01, input_file01)

    input_file02 = input_folder / flair_time02.name
    shutil.copy(flair_time02, input_file02)

    output_folder = data_folder / "output"
    output_folder.mkdir(exist_ok=True)

    precessed_folder = data_folder / "input" / "processed"
    precessed_folder.mkdir(exist_ok=True, parents=True)

    # call(
    #     [
    #         "python", "Anima-Scripts-Public/ms_lesion_segmentation/animaMSLongitudinalPreprocessing.py",
    #         "-i", input_folder,
    #         "-o", precessed_folder,
    #     ], shell=True
    # )

    # call(["python", "inference.py", CONTEXT_PATH, precessed_folder, "test_pred.nii", "--out_folder", output_folder])

    img = tio.ScalarImage(input_file01)
    random_tensor = torch.rand(img.shape)

    label_map = tio.LabelMap(tensor=(random_tensor > 0.95))
    label_map.save(output)
