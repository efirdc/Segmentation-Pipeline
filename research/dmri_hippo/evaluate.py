""" Run with `python -m research.dmri_hippo.evaluate`
"""

import json
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torchio as tio
import wandb
from PIL import Image
from segmentation_pipeline.data_processing.subject_filters import ComposeFilters, ForbidAttributes, RequireAttributes
from segmentation_pipeline.data_processing.subject_folder import SubjectFolder
from segmentation_pipeline.data_processing.subject_loaders import AttributeLoader, ComposeLoaders, ImageLoader
from segmentation_pipeline.evaluators.label_map_evaluator import LabelMapEvaluator
from segmentation_pipeline.evaluators.labeled_tensor import LabeledTensor
from segmentation_pipeline.evaluators.segmentation_evaluator import SegmentationEvaluator
from segmentation_pipeline.loggers.wandb_logger import to_wandb
from segmentation_pipeline.segmentation_trainer import ScheduledEvaluation


def load_config_files(path):

    configuration_files = glob(f"{path}/*.json")
    configs = dict()
    for config_file in configuration_files:
        stem = Path(config_file).stem
        with open(config_file, "r") as f:
            configs[stem] = json.load(f)

    return configs


def flatten(d, parent_key="", sep="__"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, to_flat_wandb(v)))
    return dict(items)


def to_flat_wandb(elem):
    if isinstance(elem, dict):
        return flatten(elem)
    elif isinstance(elem, pd.DataFrame):
        return wandb.Table(dataframe=elem)
    elif isinstance(elem, Image.Image):
        return wandb.Image(elem)
    elif isinstance(elem, LabeledTensor):
        return wandb.Table(dataframe=elem.to_dataframe())
    return elem


def get_cohorts(cohort_mode):

    cohorts = dict()
    if cohort_mode == "test":
        cohorts["cbbrain_test"] = RequireAttributes({"protocol": "cbbrain", "rescan_id": "None", 'cbbrain_test': True})
        cohorts["ab300_test"] = ComposeFilters(
            [
                RequireAttributes({"protocol": "ab300", "rescan_id": "None"}),
                ForbidAttributes({"ab300_validation": True}),
                RequireAttributes(["y"]),
            ]
        )
        cohorts["rescans"] = ForbidAttributes({"rescan_id": "None"})

        cohorts["ab300_unlabeled"] = ComposeFilters(
            [
                RequireAttributes({"protocol": "ab300", "rescan_id": "None"}),
                ForbidAttributes({"ab300_validation": True}),
                ForbidAttributes(["y"]),
            ]
        )
    elif cohort_mode == "validation":
        cohorts["cbbrain_validation"] = ComposeFilters(
            [RequireAttributes({"protocol": "cbbrain"}), RequireAttributes(["fold"])]
        )
        cohorts["ab300_validation"] = RequireAttributes({"protocol": "ab300", "ab300_validation": True})

    else:
        raise ValueError("Invalid mode provided. Must be either 'validation' or 'test'")

    return cohorts


def main(ground_truth_path: str, predictions_path: str, project: str, group: str, cohort_mode="validation"):
    subject_loader = ComposeLoaders(
        [
            ImageLoader(
                glob_pattern="whole_roi.*",
                image_name="y",
                image_constructor=tio.LabelMap,
                label_values={"left_whole": 1, "right_whole": 2},
            ),
            ImageLoader(glob_pattern="mean_dwi.*", image_name="mean_dwi", image_constructor=tio.ScalarImage),
            AttributeLoader(glob_pattern="attributes.*"),
            AttributeLoader(
                glob_pattern="../../attributes/cross_validation_split.json", multi_subject=True, uniform=True
            ),
            AttributeLoader(
                glob_pattern="../../attributes/ab300_validation_subjects.json", multi_subject=True, uniform=True
            ),
            AttributeLoader(
                glob_pattern="../../attributes/cbbrain_test_subjects.json", multi_subject=True, uniform=True
            ),
        ]
    )

    cohorts = get_cohorts(cohort_mode)

    subjects = SubjectFolder(
        root=ground_truth_path, subject_path="subjects", subject_loader=subject_loader, cohorts=cohorts
    )

    configs = load_config_files(predictions_path)

    curve_params = {
        "left_whole": np.array([-1.96312119e-01, 9.46668029e00, 2.33635173e03]),
        "right_whole": np.array([-2.68467331e-01, 1.67925603e01, 2.07224236e03]),
    }
    evaluators = [
        ScheduledEvaluation(
            evaluator=LabelMapEvaluator(
                "y_pred",
                curve_params=curve_params,
                curve_attribute="age",
                stats_to_output=("volume", "error", "absolute_error", "squared_error", "percent_diff"),
            ),
            log_name="predicted_label_eval",
            cohorts=["cbbrain_validation", "ab300_validation", "cbbrain_test", "ab300_test", "ab300_unlabeled"],
        ),
        ScheduledEvaluation(
            evaluator=SegmentationEvaluator("y_pred", "y"),
            log_name="segmentation_eval",
            cohorts=["cbbrain_validation", "cbbrain_test", "ab300_test"],
        ),
    ]

    for name, config in configs.items():
        run = wandb.init(reinit=True, project=project, config=config, group=group, name=name)

        pred_loader = ImageLoader(
            glob_pattern=f"{config['output_filename']}",
            image_name="y_pred",
            image_constructor=tio.LabelMap,
            label_values={"left_whole": 1, "right_whole": 2},
        )
        subjects.load_additional_data(Path(predictions_path) / "subjects", pred_loader)
        log_data = dict()
        for scheduled_evaluator in evaluators:
            valid_cohorts = [cohort for cohort in scheduled_evaluator.cohorts if cohort in subjects.cohorts]
            for cohort in valid_cohorts:

                cohort_subjects = subjects.cohorts[cohort](subjects.subjects)
                subjects_eval = [subject for subject in cohort_subjects if "y_pred" in subject]

                if len(cohort_subjects) > len(subjects_eval):
                    warnings.warn("Some subjects in cohort '{cohort}' are missing predictions", RuntimeWarning)

                if len(subjects_eval) > 0:
                    results = scheduled_evaluator.evaluator(subjects_eval)
                    log_data[f"{scheduled_evaluator.log_name}/{cohort}"] = results

        run.log(to_flat_wandb(log_data))
        run.log(to_wandb(log_data))
        run.finish()

        for subject in subjects:
            if 'y_pred' in subject:
                del subject["y_pred"]


if __name__ == "__main__":
    main(
        ground_truth_path="/Volumes/Extra Files/Diffusion_MRI_cropped",
        predictions_path="/Volumes/Extra Files/Predictions/EnsembleFolds",
        project="dmri-test-eval",
        group="ensembled",
        cohort_mode="test",
    )
