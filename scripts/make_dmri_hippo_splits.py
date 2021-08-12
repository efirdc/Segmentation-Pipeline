import json
import argparse
from pathlib import Path

import torch

from segmentation_pipeline import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dmri hippo splits.")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    config = load_module("../configs/diffusion_hippocampus.py")

    variables = dict(DATASET_PATH=args.dataset_path)
    context = config.get_context(device, variables)
    context.init_components()

    output_labels = ["whole_roi"]
    cbbrain_dataset = context.dataset.get_cohort_dataset(ComposeFilters([
        RequireAttributes(output_labels),
        RequireAttributes({"pathologies": "None", "rescan_id": "None"}),
        RequireAttributes({"protocol": "cbbrain"}),
    ]))
    test_filter = StratifiedFilter(size=53, continuous_attributes=['age'], discrete_attributes=['gender'],
                                   seed=args.seed)
    cbbrain_test_dataset = cbbrain_dataset.get_cohort_dataset(test_filter)
    cbbrain_cross_validation_dataset = cbbrain_dataset.get_cohort_dataset(NegateFilter(test_filter))

    assert len(cbbrain_test_dataset) == 53
    assert len(cbbrain_cross_validation_dataset) == 100

    num_test_male = len([subject for subject in cbbrain_test_dataset.subjects
                    if subject['gender'] == 'M'])
    ages = [subject['age'] for subject in cbbrain_test_dataset.subjects]
    ages.sort()
    print(f"Testing males: {num_test_male}, females: {53 - num_test_male}")
    print(f"Testing ages: {ages}")

    cross_validation_fold_ids = random_folds(len(cbbrain_cross_validation_dataset), num_folds=5,
                                             seed=args.seed)

    ab300_validation_dataset = context.dataset.get_cohort_dataset(ComposeFilters([
        ForbidAttributes(output_labels),
        RequireAttributes({"pathologies": "None", "rescan_id": "None"}),
        RequireAttributes({"protocol": "ab300"}),
        StratifiedFilter(size=50, continuous_attributes=['age'], discrete_attributes=['gender'], seed=args.seed)
    ]))

    assert len(ab300_validation_dataset) == 50

    dataset_path = Path(args.dataset_path)
    with open(dataset_path / "attributes" / "cbbrain_test_subjects.json", "w") as f:
        out_dict = {
            subject['name']: {'cbbrain_test': True}
            for subject in cbbrain_test_dataset.subjects
        }
        json.dump(out_dict, f, indent=4)

    with open(dataset_path / "attributes" / "ab300_validation_subjects.json", "w") as f:
        out_dict = {
            subject['name']: {'ab300_validation': True}
            for subject in ab300_validation_dataset.subjects
        }
        json.dump(out_dict, f, indent=4)

    with open(dataset_path / "attributes" / "cross_validation_split.json", "w") as f:
        out_dict = {
            subject['name']: {'fold': fold_id}
            for subject, fold_id in zip(cbbrain_cross_validation_dataset.subjects, cross_validation_fold_ids)
        }
        json.dump(out_dict, f, indent=4)
