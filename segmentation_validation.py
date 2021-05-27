from torch.utils.data import Dataset
from typing import Dict, Sequence
from utils import slice_volume
import torchio as tio
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

import warnings
from PIL import Image
import io


class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, subjects: Sequence[tio.Subject]):
        pass


class ContourImageEvaluator(Evaluator):
    def __init__(
            self,
            name: str,
            plane: str,
            image_channel_id: int,
            slice_id: int,
            legend: bool,
            ncol: int,
            subject_names: Sequence[str]
    ):
        self.name = name
        self.plane = plane
        self.image_channel_id = image_channel_id
        self.slice_id = slice_id
        self.legend = legend
        self.ncol = ncol
        self.subject_names = subject_names

    def evaluate(self, subjects):
        imgs = []
        ys = []
        y_preds = []
        for subject in subjects:
            imgs.append(slice_volume(subject["X"], self.image_channel_id, self.plane, self.slice_id).unsqueeze(0))
            ys.append(slice_volume(subject["y"], 0, self.plane, self.slice_id).unsqueeze(0))
            y_preds.append(slice_volume(subject["y_pred"], 0, self.plane, self.slice_id).unsqueeze(0))
        img, y, y_pred = [
            make_grid(x, nrow=self.ncol, pad_value=-1, padding=1)[0].cpu()
            for x in (imgs, ys, y_preds)
        ]
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


        for name, label_id in self.label_names.items():
            contour = plt.contour(X_grid, Y_grid, y == label_id, levels=[0.5], colors=cmap[label_id:label_id+1],
                                  **options)
            plt.contour(X_grid, Y_grid, y_pred == label_id, levels=[0.95], linestyles="dashed",
                        colors=cmap[label_id:label_id+1], **options)
            contours.append(contour)
        if self.legend:
            plt.legend([contour.legend_elements()[0][0] for contour in contours], label_names_inverse.items(),
                       ncol=3, bbox_to_anchor=(0.5, 0), loc='upper center', fancybox=True)
        warnings.resetwarnings()
        plt.tick_params(which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight", pad_inches=0.0, facecolor="black")
        buf.seek(0)
        pil_image = Image.open(buf)
        plt.close(fig)
        out = {self.name: pil_image}
        return out


class VolumeStatsEvaluator(Evaluator):



class ValidationDataset:
    def __init__(
            self,
            name: str,
    ):


class SegmentationEvaluator:
    def __init__(
            self,
            val_datasets: Sequence[Dataset],
            val_images: Sequence[ValidationImage]
    ):
        self.val_datasets = val_datasets
        self.val_image = val_images