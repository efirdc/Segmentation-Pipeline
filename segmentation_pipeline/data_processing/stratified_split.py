"""Generate stratifed test train splits and adds attribute isTrain to the subject. 
Run with `python -m segmentation_pipeline.data_processing.stratified_split`

Update the ROOT_PATH variable to the path of the data
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torchio as tio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from ..data_processing import *

np.random.seed(5)
ROOT_PATH = Path("")

discrete_attributes = ["gender"]
continuous_attributes = ["age"]

split_attributes = discrete_attributes + continuous_attributes

subject_loader = ComposeLoaders(
    [
        ImageLoader(glob_pattern="mean_dwi.*", image_name="mean_dwi", image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="md.*", image_name="md", image_constructor=tio.ScalarImage),
        ImageLoader(glob_pattern="fa.*", image_name="fa", image_constructor=tio.ScalarImage),
        ImageLoader(
            glob_pattern="whole_roi.*",
            image_name="whole_roi",
            image_constructor=tio.LabelMap,
            label_values={"left_whole": 1, "right_whole": 2},
        ),
        AttributeLoader(glob_pattern="attributes.*"),
    ]
)

output_labels = ["whole_roi"]
input_images = ["mean_dwi", "md", "fa"]

cohorts = dict()
cohorts["all"] = RequireAttributes(input_images)
cohorts["cbbrain"] = ComposeFilters(
    [
        RequireAttributes(output_labels),
        RequireAttributes({"pathologies": "None", "rescan_id": "None"}),
        RequireAttributes({"protocol": "cbbrain"}),
    ]
)
cohorts["ab300"] = ComposeFilters(
    [
        ForbidAttributes(output_labels),
        RequireAttributes({"pathologies": "None", "rescan_id": "None"}),
        RequireAttributes({"protocol": "ab300"}),
    ]
)


subjects = SubjectFolder(root=ROOT_PATH, subject_path="subjects", subject_loader=subject_loader, cohorts=cohorts)


cbbrain_subjects = subjects.get_cohort_dataset("cbbrain")
ab300_subjects = subjects.get_cohort_dataset("ab300")


def get_splits(subjects, test_size):
    df = pd.DataFrame(columns=["name", "subject_id"] + split_attributes)

    for subject in subjects:
        subject_dict = dict()
        subject_dict["name"] = subject["name"]
        subject_dict["subject_id"] = subject["subject_id"]
        for attribute in split_attributes:
            subject_dict[attribute] = subject[attribute]
        df = df.append(subject_dict, ignore_index=True)

    for continuous_attribute in continuous_attributes:
        discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
        df[continuous_attribute] = discretizer.fit_transform(
            df[continuous_attribute].to_numpy().reshape(-1, 1)
        ).reshape(-1)

    train, test = train_test_split(df, test_size=test_size, stratify=df[split_attributes])
    return train, test


cb_train, cb_test = get_splits(cbbrain_subjects, 53)
ab_train, ab_test = get_splits(ab300_subjects, 50)
subjects_path = ROOT_PATH / "subjects"
for sub_path in subjects_path.iterdir():

    with open(sub_path / "attributes.json", "r") as f:
        attributes = json.load(f)

    if (cb_train["name"] == sub_path.name).any() or (ab_train["name"] == sub_path.name).any():
        attributes["isTrain"] = True
    else:
        attributes["isTrain"] = False

    with open(sub_path / "attributes.json", "w") as f:
        json.dump(attributes, f, indent=4)


