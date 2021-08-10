import os
import json
import copy

from ..transforms import CustomSequentialLabels


def save_dataset_as_nn_unet(dataset, output_path, short_name, image_names, label_map_name,
                            train_cohort, test_cohort=None, metadata=None, fix_affine=False):
    train_image_path = os.path.join(output_path, 'imagesTr')
    train_label_path = os.path.join(output_path, 'labelsTr')
    test_image_path = os.path.join(output_path, 'imagesTs')
    for folder in (train_image_path, train_label_path, test_image_path):
        if not os.path.exists(folder):
            os.makedirs(folder)

    train_dataset = dataset.get_cohort_dataset(train_cohort)

    def save_images(image_path, subject_id, subject):
        channel_id = 0
        for image_name in image_names:
            image = subject[image_name]
            if fix_affine:
                image.affine = image.affine

            for image_channel in image.data.split(1):
                out_image = copy.deepcopy(image)
                out_image.set_data(image_channel)

                out_file_name = f'{short_name}_{subject_id:03}_{channel_id:04}.nii.gz'
                out_image.save(os.path.join(image_path, out_file_name))

                channel_id += 1

    subject_id = 0
    for subject in train_dataset.all_subjects:
        subject_id += 1

        assert all(image_name in subject for image_name in image_names)
        assert label_map_name in subject

        save_images(train_image_path, subject_id, subject)

        label_map = subject[label_map_name]
        label_map = CustomSequentialLabels()(label_map)

        if fix_affine:
            label_map.affine = subject[image_names[0]].affine

        out_file_name = f"{short_name}_{subject_id:03}.nii.gz"
        label_map.save(os.path.join(train_label_path, out_file_name))

    if test_cohort is not None:
        test_dataset = dataset.get_cohort_dataset(test_cohort)

        for subject in test_dataset.all_subjects:
            subject_id += 1

            assert all(image_name in subject for image_name in image_names)

            save_images(test_image_path, subject_id, subject)

    label_values = train_dataset.all_subjects[0][label_map_name]['label_values']
    label_values = {"background": 0, **label_values}

    if metadata is None:
        metadata = {}

    json_dict = {
        'name': short_name,
        **({} if metadata is None else metadata),
        'tensorImageSize': "4D",
        "modality": {str(i): image_name for i, image_name in enumerate(image_names)},
        "labels": {str(label_value): label_name for label_name, label_value in label_values.items()},
        "numTraining": len(train_dataset),
        "numTest": len(test_dataset) if test_cohort is not None else 0,
        "training": [
            {
                "image": f'./imagesTr/{short_name}_{i:03}.nii.gz',
                "label": f'./labelsTr/{short_name}_{i:03}.nii.gz'
            }
            for i in range(1, len(train_dataset) + 1)
        ],
        "test": [] if test_cohort is None else [
            f"./imagesTs/{short_name}_{i:03}.nii.gz"
            for i in range(len(train_dataset) + 1, len(train_dataset) + len(test_dataset) + 1)
        ]
    }

    json_path = os.path.join(output_path, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
