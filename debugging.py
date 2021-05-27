import torch
from utils import load_module


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    config = load_module("./configs/diffusion_hippocampus.py")

    variables = dict(DATASET_PATH="X:/Datasets/Diffusion_MRI/Subjects/", CHECKPOINTS_PATH="X:/Checkpoints/")
    context = config.get_context(device, variables)

    print("Collate fn:", context.dataloader.collate_fn)

    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch

    for i in range(2):
        loader = sample_data(context.dataloader)
        batch = next(loader)
        print(batch)