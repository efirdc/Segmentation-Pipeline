import torch
from utils import load_module
import torch
import torchio as tio
from torch.utils.data import DataLoader

if __name__ == "__main__":
    config = load_module("./configs/msseg2.py")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    variables = dict(DATASET_PATH="X:/Datasets/MSSEG2_processed/", CHECKPOINTS_PATH="X:/Checkpoints/")
    context = config.get_context(device, variables)
    context.init_components()

    patch_size = 96
    queue_length = 300
    samples_per_volume = 10

    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        context.dataset,
        queue_length,
        samples_per_volume,
        sampler,
        num_workers=0,
    )

    patches_loader = DataLoader(dataset=patches_queue, batch_size=2, collate_fn=context.dataset.collate)

    num_epochs = 2
    for epoch_index in range(num_epochs):
        for patches_batch in patches_loader:
            print(patches_batch)
