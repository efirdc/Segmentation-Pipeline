import io
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from PIL import Image
from torchvision.utils import make_grid

from utils import slice_volume
from .evaluator import Evaluator


class ContourImageEvaluator(Evaluator):
    def __init__(
            self,
            name: str,
            plane: str,
            image_channel_id: int,
            slice_id: int,
            legend: bool,
            ncol: int,
    ):
        self.name = name
        self.plane = plane
        self.image_channel_id = image_channel_id
        self.slice_id = slice_id
        self.legend = legend
        self.ncol = ncol

    def __call__(self, subjects):
        imgs = []
        ys = []
        y_preds = []
        for subject in subjects:
            imgs.append(slice_volume(subject["X"].data, self.image_channel_id, self.plane, self.slice_id).unsqueeze(0))
            ys.append(slice_volume(subject["y"].data, 0, self.plane, self.slice_id).unsqueeze(0))
            y_preds.append(slice_volume(subject["y_pred"].data, 0, self.plane, self.slice_id).unsqueeze(0))
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
            plt.legend([contour.legend_elements()[0][0] for contour in contours], subject[0]["y"]['label_names'].items(),
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
