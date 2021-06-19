import torch
from utils import load_module


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    #config = load_module("./configs/diffusion_hippocampus.py")
    #variables = dict(DATASET_PATH="X:/Datasets/Diffusion_MRI/Subjects/", CHECKPOINTS_PATH="X:/Checkpoints/")
    #context = config.get_context(device, variables)

    config = load_module("./configs/qsm_deep_grey_matter.py")
    variables = dict(DATASET_PATH="X:/Datasets/DGM/segmentation_3T_ps18_v3/", CHECKPOINTS_PATH="X:/Checkpoints/")
    context = config.get_context(device, variables)

    untransformed_subject = context.dataset.subjects[0]
    print("Original labels:")
    print(untransformed_subject.dgm['label_values'])

    subject = context.dataset[0]
    print("\nTransformed labels:")
    print(subject.dgm['label_values'])

    inverse_subject = subject.apply_inverse_transform(warn=False)
    print("\nInverse transformed labels:")
    print(inverse_subject.dgm['label_values'])

    '''
    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch

    for i in range(2):
        loader = sample_data(context.dataloader)
        batch = next(loader)
        print(batch)
    '''