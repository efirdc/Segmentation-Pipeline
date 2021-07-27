import time
import argparse
import tarfile
import shutil
from pathlib import Path
import torch

from loggers import *
from torch_context import TorchContext
from utils import load_module


# Converts a time string with format days-hours:minutes:seconds to the number of seconds (integer)
# i.e. time_str_to_seconds("2-3:30:5") returns 185405, the number of seconds in 2 days, 3 hours, 30 minutes, 5 seconds
def time_str_to_seconds(timestr):
    dash_split = timestr.split("-")
    assert len(dash_split) == 2
    D = int(dash_split[0])
    HMS = dash_split[1]

    colon_split = [int(x) for x in HMS.split(":")]
    assert len(colon_split) == 3
    H, M, S = colon_split

    return ((D * 24 + H) * 60 + M) * 60 + S


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Network Trainer")
    parser.add_argument("dataset_path", type=str,
                        help="Path to the dataset. If the dataset is a .tar file it will be unzipped first."
                        )
    parser.add_argument("config", type=str, default="",
                        help="Path to a python file which sets up the configuration. This file must define a function"
                             "get_context() which returns a TorchContext. See the ./configs/ folder for examples."
                        )
    parser.add_argument("--work_path", type=str, default=None,
                        help="Copy the dataset to this directory before training begins."
                        )
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to a training checkpoint to resume training."
                        )
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Number of training iterations. "
                             "Training may stop early if the training time expires first."
                        )
    parser.add_argument("--training_time", type=str, default="28-00:00:00",
                        help="Length of time to train for. Format: days-hours:minutes:seconds "
                             "Training may stop early if the number of iterations run out first."
                        )
    parser.add_argument("--device", type=str, default="cuda",
                        help="Length of time to train for. Format: days-hours:minutes:seconds "
                             "Training may stop early if the number of iterations run out first."
                        )
    parser.add_argument("--logger", type=str, default=None,
                        help="Logging implementation to use. Set to 'wandb' to enable Weights and Bias logging service."
                        )
    parser.add_argument("--project_name", type=str, default="segmentation",
                        help="Project name for logger."
                        )
    parser.add_argument("--logging_dir", type=str, help="Path to directory for saving checkpoints.")
    parser.add_argument("--group_name", type=str, default=None,
                        help="Optional name for grouping runs."
                        )
    parser.add_argument("--preload_training_data", type=bool, default=False,
                        help="Optionally preload the entire training dataset into memory."
                        )
    parser.add_argument("--preload_validation_data", type=bool, default=False,
                        help="Optionally preload the entire validation dataset into memory."
                        )
    parser.add_argument("--num_workers", type=int, default=0,
                        help="How many CPU threads to use for data loading, preprocessing, and augmentation."
                        )
    parser.add_argument("--validation_batch_size", type=int, default=4,
                        help="How many validation subjects should be run through the model at once."
                        )
    parser.add_argument("--patch_queue_length", type=int, default=100,
                        help="How many volumes should be loaded into the queue for patch based training."
                        )
    parser.add_argument("--force_continue", type=bool, default=False,
                        help="Force the model to continue training by resetting the max score on the trainer."
                        )
    args, unknown_args = parser.parse_known_args()

    # Parse extra unknown keyword arguments that were passed on the command line
    assert len(unknown_args) % 2 == 0, f"Extra arguments have an invalid pairing: {unknown_args}"
    extra_args = {}
    for i in range(0, len(unknown_args), 2):
        name = unknown_args[i]
        value = unknown_args[i + 1]
        if value.isdigit():
            value = int(value)
        elif value.isnumeric():
            value = float(value)
        assert name.startswith('--'), f"Extra arguments must start with '--', found: {name}"
        extra_args[name[2:]] = value

    # Compute time when training should terminate.
    # A buffer is used for saving the model (either 5 minutes or 10% of training time, whichever is smaller)
    training_time = time_str_to_seconds(args.training_time)
    save_buffer = min(int(training_time * 0.1), 5 * 60)
    stop_time = time.time() + training_time - save_buffer

    # Extract the dataset if it is a .tar file and copy it to --work_dir if it was specified
    dataset_path = Path(args.dataset_path)
    if dataset_path.is_file():
        assert dataset_path.suffix == '.tar', f"Dataset file extension must be .tar not {dataset_path.suffix}"
        if args.work_path is None:
            extract_dir = dataset_path.parent
        else:
            extract_dir = Path(args.work_path)
        extract_dir_contents = [child.stem for child in list(extract_dir.iterdir())]
        with tarfile.open(name=dataset_path, mode="r") as tar:
            first_tar_file = tar.getnames()[0]
            if first_tar_file in extract_dir_contents:
                print(f"Dataset already extracted to {extract_dir}")
            else:
                print(f"Extracting {dataset_path} to {extract_dir}")
                tar.extractall(extract_dir)
        dataset_path = extract_dir
        contents = list(dataset_path.iterdir())
        if len(contents) == 1:
            dataset_path = contents[0]
    elif args.work_path is not None:
        work_path = Path(args.work_path + "/" + dataset_path.stem)
        if work_path.exists():
            print(f"Dataset already transfered to {work_path}")
        else:
            print(f"Copying dataset from {dataset_path} to {work_path}")
            shutil.copytree(dataset_path, work_path)
        dataset_path = work_path
    print(f"Using dataset path {dataset_path}")

    # Get device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device {device}")

    # Initialize a new context, or load from a file to resume training
    variables = dict(DATASET_PATH=str(dataset_path))
    if args.load_checkpoint is None:
        config = load_module(args.config)
        context = config.get_context(device, variables, **extra_args)
    else:
        context = TorchContext(device, file_path=args.load_checkpoint, variables=variables)
    context.init_components()

    if args.force_continue:
        context.trainer.max_score = float('-inf')
        context.trainer.max_score_iteration = context.trainer.iteration

    if args.logger == 'wandb':
        logger = WandbLogger(args.project_name, args.logging_dir, group_name=args.group_name)
    else:
        logger = NonLogger()

    print("entering training loop")
    context.trainer.train(context,
                          args.iterations,
                          stop_time=stop_time,
                          preload_training_data=args.preload_training_data,
                          preload_validation_data=args.preload_validation_data,
                          num_workers=args.num_workers,
                          validation_batch_size=args.validation_batch_size,
                          validation_patch_batch_size=args.validation_batch_size,
                          patch_queue_length=args.patch_queue_length,
                          logger=logger)
