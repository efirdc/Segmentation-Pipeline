import math
import os
import random
from typing import Sequence

import torch
import torchio as tio
import matplotlib.pyplot as plt

from ..evaluators import ContourImageEvaluator


def vis_features(x):
    from ipywidgets import interact

    N, C, W, H, D = x.shape

    @interact(i=(0, N-1), c=(0, C-1), d=(0, D-1))
    def plot_feature_map(i, c, d):
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(x[i, c, :, :, d].cpu(), cmap="gray")
        plt.colorbar()
        plt.show()
        plt.close(fig)


def vis_subject(context, subject):
    from ipywidgets import interact

    if isinstance(subject, Sequence):
        subjects = subject
        subject = subject[0]
    elif isinstance(subject, tio.Subject):
        subjects = [subject]

    images = {key: val for key, val in subject.items() if isinstance(val, tio.ScalarImage)}
    label_maps = {key: val for key, val in subject.items() if isinstance(val, tio.LabelMap)}

    @interact(image_name=images.keys(),
              target_label_map_name=[None, *label_maps.keys()],
              prediction_label_map_name=[None, *label_maps.keys()],
              plane=['Axial', 'Coronal', 'Saggital', 'interesting', 'random'])
    def select_images(image_name, target_label_map_name, prediction_label_map_name, plane):
        image = images[image_name]

        W, H, D = image.spatial_shape
        if plane == 'random':
            plane = ('Axial', 'Coronal', 'Saggital')[random.randint(0, 2)]
        num_slices = {'Axial': D, 'Coronal': H, 'Saggital': W, 'interesting': 20}[plane]

        @interact(save=False, legend=True, ticks=False, scale=(0.05, 0.15, 0.01),
                  line_width=(0.5, 2.5), slice_id=(0, num_slices-1), interesting_slice=False)
        def select_slice(save, legend, ticks, scale, line_width, slice_id, interesting_slice):

            evaluator = ContourImageEvaluator(
                plane=plane, image_name=image_name,
                target_label_map_name=target_label_map_name,
                prediction_label_map_name=prediction_label_map_name,
                slice_id=slice_id, legend=legend, ncol=int(math.sqrt(len(subjects))),
                scale=scale, line_width=line_width,
                interesting_slice=interesting_slice
            )

            pil_image = evaluator(subjects)
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(pil_image)
            if not ticks:
                plt.tick_params(which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
            if save:
                save_dir = f"./images/{context.name}/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                params = [subject['name'], image_name, plane, slice_id,
                          target_label_map_name, prediction_label_map_name]
                params = [str(p) for p in params if p is not None]

                file_name = f"{'-'.join(params)}.png"
                fig.savefig(save_dir + file_name, bbox_inches="tight", pad_inches=0.0, facecolor="black")
            plt.show()
            plt.close(fig)


def vis_model(context, subject):
    from ipywidgets import interact

    X = subject['X']['data'].unsqueeze(0).to(context.device)
    modules = list(context.model.named_modules())

    @interact(module=modules[1:])
    def select_module(module):

        def forward_module_hook(module, x_in, x_out):
            vis_features(x_out.cpu())

        hook_handle = module.register_forward_hook(forward_module_hook)
        with torch.no_grad():
            context.model(X)
        hook_handle.remove()
