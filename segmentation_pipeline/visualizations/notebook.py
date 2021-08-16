import math
import os
import random
from typing import Sequence

import torch
import torchio as tio
from ipywidgets import interact
import matplotlib.pyplot as plt

from ..evaluators import ContourImageEvaluator


def vis_features(x):
    N, C, W, H, D = x.shape

    @interact(i=(0, N-1), c=(0, C-1), d=(0, D-1))
    def plot_feature_map(i, c, d):
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(x[i, c, :, :, d].cpu(), cmap="gray")
        plt.colorbar()
        plt.show()
        plt.close(fig)


def vis_subject(context, subject):
    if isinstance(subject, Sequence):
        subjects = subject
        subject = subject[0]
    elif isinstance(subject, tio.Subject):
        subjects = [subject]

    images = {key: val for key, val in subject.items() if isinstance(val, tio.ScalarImage)}
    label_maps = {key: val for key, val in subject.items() if isinstance(val, tio.LabelMap)}

    @interact(image_name=images.keys(),
              label_map_name=label_maps.keys(),
              plane=['Axial', 'Coronal', 'Saggital', 'interesting', 'random'])
    def select_images(image_name, label_map_name, plane):
        label_map_name = 'y'
        image = images[image_name]
        label_map = label_maps[label_map_name]
        W, H, D = image.spatial_shape
        if plane == 'random':
            plane = ('Axial', 'Coronal', 'Saggital')[random.randint(0, 2)]
        num_slices = {'Axial': D, 'Coronal': H, 'Saggital': W, 'interesting': 20}[plane]

        @interact(save=False, show_labels=True, legend=True, ticks=False, scale=(0.05, 0.15, 0.01),
                  line_width=(0.5, 2.5), slice_id=(0, num_slices-1), interesting_slice=False)
        def select_slice(save, show_labels, legend, ticks, scale, line_width, slice_id, interesting_slice):

            if 'y_pred' in subject:
                prediction_label_map_name = 'y_pred'
            else:
                prediction_label_map_name = None

            evaluator = ContourImageEvaluator(
                plane=plane, image_name=image_name,
                target_label_map_name=label_map_name if show_labels else None,
                prediction_label_map_name=prediction_label_map_name if show_labels else None,
                slice_id=slice_id, legend=legend, ncol=int(math.sqrt(len(subjects))),
                scale=scale, line_width=line_width,
                interesting_slice=interesting_slice
            )

            pil_image = evaluator(subjects)
            fig = plt.figure(figsize=(14, 14))
            plt.imshow(pil_image)
            if not ticks:
                plt.tick_params(which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
            if save:
                save_dir = f"./images/{context.name}/iter{context.iteration:08}/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                file_name = f"{subject['name']}_{image_name}_{plane}_{slice_id}.png"
                fig.savefig(save_dir + file_name, bbox_inches="tight", pad_inches=0.0, facecolor="black")
            plt.show()
            plt.close(fig)


def vis_model(context, subject):
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
