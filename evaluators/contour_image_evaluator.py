import io
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from PIL import Image
from torchvision.utils import make_grid

from utils import slice_volume
from .evaluator import Evaluator


class ContourImageEvaluator(Evaluator):
    def __init__(
            self,
            plane: str,
            image_name: str,
            prediction_label_map_name: str,
            target_label_map_name: str,
            slice_id: int,
            legend: bool,
            ncol: int,
    ):
        self.plane = plane
        self.image_name = image_name
        self.prediction_label_map_name = prediction_label_map_name
        self.target_label_map_name = target_label_map_name
        self.slice_id = slice_id
        self.legend = legend
        self.ncol = ncol

    def slice_and_make_grid(self, subjects, image_name, channel, impute_shape, pad_value=0):
        slices = []
        for subject in subjects:
            if image_name in subject:
                slices.append(slice_volume(subject[image_name].data, channel, self.plane, self.slice_id).unsqueeze(0))
            else:
                slices.append(torch.zeros(impute_shape))

        return make_grid(slices, nrow=self.ncol, pad_value=pad_value, padding=1)[0].cpu()

    def __call__(self, subjects):
        out_pred = self.prediction_label_map_name is not None
        out_target = self.target_label_map_name is not None

        if out_pred:
            label_values = subjects[0][self.prediction_label_map_name]['label_values']
        if out_target:
            label_values = subjects[0][self.target_label_map_name]['label_values']

        sample_slice = slice_volume(subjects[0][self.image_name].data, 0, self.plane, self.slice_id)
        impute_shape = sample_slice.shape

        img = self.slice_and_make_grid(subjects, self.image_name, 0, impute_shape, pad_value=-1)
        if out_target:
            y = {
                label_name: self.slice_and_make_grid(subjects, self.target_label_map_name, label_value, impute_shape).bool()
                for label_name, label_value in label_values.items()
            }
        if out_pred:
            y_pred = {
                label_name: self.slice_and_make_grid(subjects, self.prediction_label_map_name, label_value, impute_shape).bool()
                for label_name, label_value in label_values.items()
            }

        H, W = img.shape
        fig = plt.figure(figsize=tuple(np.array((W, H)) / 7.))
        plt.imshow(img, cmap="gray", vmin=-1, vmax=1)
        X_grid, Y_grid = np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H))
        options = dict(linewidths=1.5, alpha=1.)
        warnings.filterwarnings("ignore")
        cmap = [None, "r", "g", "b", "y", "c", "m"] \
               + list(get_cmap("Accent").colors) + list(get_cmap("Dark2").colors) \
               + list(get_cmap("Set1").colors) + list(get_cmap("Set2").colors) \
               + list(get_cmap("tab20").colors)
        contours = []

        for label_name, label_id in label_values.items():
            contour = plt.contour(X_grid, Y_grid, y[label_name], levels=[0.5], colors=cmap[label_id:label_id+1],
                                  **options)
            plt.contour(X_grid, Y_grid, y_pred[label_name], levels=[0.95], linestyles="dashed",
                        colors=cmap[label_id:label_id+1], **options)
            contours.append(contour)
        if self.legend:
            plt.legend(
                [contour.legend_elements()[0][0] for contour in contours],
                label_values.items(), ncol=3, bbox_to_anchor=(0.5, 0), loc='upper center', fancybox=True
            )
        warnings.resetwarnings()
        plt.tick_params(which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight", pad_inches=0.0, facecolor="black")
        buf.seek(0)
        pil_image = Image.open(buf)
        plt.close(fig)
        out = {"image": pil_image}
        return out
