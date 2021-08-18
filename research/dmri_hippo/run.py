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
        validation_batch_size=4,
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
        preload_training_data=False,
        preload_validation_data=False,
        num_workers=num_cpu_threads,
        validation_batch_size=16,
        logger=WandbLogger("dmri-hippo-seg-v3", logging_path)
    )


if __name__ == "__main__":
    fire.Fire({
        "main": main,
        "debug": debug,
        "augmentation_experiment": augmentation_experiment,
    })
