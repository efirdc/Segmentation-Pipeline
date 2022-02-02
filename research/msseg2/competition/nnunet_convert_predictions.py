import shutil
from pathlib import Path

from segmentation_pipeline import *


def main(
        raw_path: PathLike,
        postprocessed_path: PathLike,
        predictions_path: PathLike,
):
    prediction_file_paths = list(predictions_path.iterdir())
    raw_file_paths = [path for path in raw_path.iterdir()
                      if path.name.endswith("nii.gz")]
    postprocessed_file_paths = [path for path in postprocessed_path.iterdir()
                                if path.name.endswith("nii.gz")]

    for pred_folder, raw_file_path, postprocessed_file_path in zip(prediction_file_paths,
                                                                   raw_file_paths,
                                                                   postprocessed_file_paths):
        shutil.copy(raw_file_path, pred_folder / "nnunet_raw.nii.gz")
        shutil.copy(raw_file_path, pred_folder / "nnunet_postprocessed.nii.gz")


if __name__ == "__main__":
    root_path = Path("X:\\Predictions\\nnUNet_trained_models\\nnUNet\\3d_fullres\\Task510_MSSEG2\\nnUNetTrainerV2__nnUNetPlansv2.1\\")
    main(
        raw_path=root_path / "cv_niftis_raw",
        postprocessed_path=root_path / "cv_niftis_postprocessed",
        predictions_path=Path("X:\\Predictions\\MSSEG2\\"),
    )