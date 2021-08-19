""" Run with `python -m research.dmri_hippo.evaluate`
"""

import json
from glob import glob
from pathlib import Path

import torchio as tio
import wandb

from segmentation_pipeline.data_processing.subject_folder import SubjectFolder
from segmentation_pipeline.data_processing.subject_loaders import ImageLoader
from segmentation_pipeline.evaluators.segmentation_evaluator import SegmentationEvaluator


def load_config_files(path):

    configuration_files = glob(f"{path}/*.json")
    configs = dict()
    for config_file in configuration_files:
        stem = Path(config_file).stem
        with open(config_file, "r") as f:
            configs[stem] = json.load(f)

    return configs


def main(ground_truth_path: str, predictions_path: str, project: str, group: str):
    subject_loader = ImageLoader(
        glob_pattern="whole_roi.*",
        image_name="y",
        image_constructor=tio.LabelMap,
        label_values={"left_whole": 1, "right_whole": 2},
    )

    subjects = SubjectFolder(root=ground_truth_path, subject_path="subjects", subject_loader=subject_loader)

    configs = load_config_files(predictions_path)

    evaluator = SegmentationEvaluator("y_pred", "y")
    for name, config in configs.items():
        run = wandb.init(reinit=True, project=project, config=config, group=group, name=name)

        pred_loader = ImageLoader(
            glob_pattern=f"{config['output_filename']}",
            image_name="y_pred",
            image_constructor=tio.LabelMap,
            label_values={"left_whole": 1, "right_whole": 2},
        )

        subjects.load_additional_data(Path(predictions_path) / "subjects", pred_loader)
        subjects_eval = [subject for subject in subjects.subjects if "y_pred" in subject]
        results = evaluator(subjects_eval)

        subjects_table = wandb.Table(
            columns=list(results["subject_stats"].columns), data=results["subject_stats"].values.tolist()
        )
        wandb.log({"subject_stats": subjects_table})

        # not sure how to log summary results

        # remove y_pred images for next run
        for subject in subjects.subjects:
            if "y_pred" in subject:
                del subject["y_pred"]

        run.finish()
if __name__ == "__main__":
    main("/Volumes/Extra Files/Diffusion_MRI/", "/Volumes/Extra Files/HippoPredictions", "hippo-eval", "run3")
