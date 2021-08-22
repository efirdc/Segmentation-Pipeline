from itertools import product

import torch
import fire

from segmentation_pipeline import prepare_dataset_files, SegmentationTrainer, WandbLogger

from .configs import main_config, augmentation


def main(
        dataset_path: str,
        logging_path: str,
        work_path: str = None,
        fold: int = 0,
        max_training_time: str = None,
        device: str = 'cuda',
):
    dataset_path = prepare_dataset_files(dataset_path, work_path)
    context = main_config.get_context(
        device=torch.device(device),
        variables={"DATASET_PATH": str(dataset_path)},
        fold=fold,
        predict_hbt=False,
    )
    context.init_components()

    trainer: SegmentationTrainer = context.trainer
    trainer.train(
        context=context,
        max_iterations=100000,
        max_training_time=max_training_time,
        preload_training_data=False,
        preload_validation_data=False,
        num_workers=4,
        validation_batch_size=16,
        logger=WandbLogger("dmri-hippo-seg-v3", logging_path)
    )


def debug(
        dataset_path: str,
        logging_path: str,
        work_path: str = None,
        fold: int = 0,
        max_training_time: str = None,
        device: str = 'cuda',
):
    dataset_path = prepare_dataset_files(dataset_path, work_path)
    context = augmentation.get_context(
        device=torch.device(device),
        variables={"DATASET_PATH": str(dataset_path)},
        augmentation_mode='combined',
        fold=fold,
        predict_hbt=False,
        training_batch_size=1,
    )
    context.init_components()

    trainer: SegmentationTrainer = context.trainer
    trainer.train(
        context=context,
        max_iterations=100000,
        max_training_time=max_training_time,
        preload_training_data=False,
        preload_validation_data=False,
        num_workers=0,
        validation_batch_size=1,
        logger=WandbLogger("dmri-hippo-seg-debugging", logging_path)
    )


def augmentation_experiment(
        dataset_path: str,
        logging_path: str,
        work_path: str = None,
        augmentation_mode: str = "no_augmentation",
        fold: int = 0,
        max_training_time: str = None,
        device: str = 'cuda',
        num_cpu_threads: int = 4,
        group_name: str = None,
):
    dataset_path = prepare_dataset_files(dataset_path, work_path)
    context = augmentation.get_context(
        device=torch.device(device),
        variables={"DATASET_PATH": str(dataset_path)},
        augmentation_mode=augmentation_mode,
        fold=fold,
        predict_hbt=False,
    )
    context.init_components()

    trainer: SegmentationTrainer = context.trainer
    trainer.train(
        context=context,
        max_iterations=100000,
        max_training_time=max_training_time,
        preload_training_data=True,
        preload_validation_data=True,
        num_workers=num_cpu_threads,
        validation_batch_size=16,
        logger=WandbLogger("dmri-hippo-seg-v3", logging_path, group_name=group_name)
    )


def augmentation_experiment_grid(
        dataset_path: str,
        logging_path: str,
        work_path: str = None,
        task_id: int = 0,
):
    grid_params = {
        "augmentation_mode": ["no_augmentation", "standard", "dwi_reconstruction", "combined"],
        "fold": range(0, 5)
    }

    configs = [
        dict(zip(grid_params.keys(), values))
        for values in product(*grid_params.values())
    ]
    config = configs[task_id]

    augmentation_experiment(
        dataset_path=dataset_path,
        logging_path=logging_path,
        work_path=work_path,
        **config,
        max_training_time=None,
        device='cuda',
        num_cpu_threads=8,
        group_name="augmentation_experiment_03",
    )


if __name__ == "__main__":
    fire.Fire({
        "main": main,
        "debug": debug,
        "augmentation_experiment": augmentation_experiment,
        "augmentation_experiment_grid": augmentation_experiment_grid
    })
