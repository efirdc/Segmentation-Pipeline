import io
import random
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from PIL import Image
from torchvision.utils import make_grid

from .evaluator import Evaluator
from ..transforms import FindInterestingSlice
from ..utils import slice_volume


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
            scale: float = 0.1,
            line_width: float = 1.5,
            interesting_slice: bool = False,
            split_subjects: bool = False,
    ):
        self.plane = plane
        self.image_name = image_name
        self.prediction_label_map_name = prediction_label_map_name
        self.target_label_map_name = target_label_map_name
        self.slice_id = slice_id
        self.legend = legend
        self.ncol = ncol
        self.scale = scale
        self.line_width = line_width
        self.interesting_slice = interesting_slice
        self.split_subjects = split_subjects

    def get_slice_id(self, subject, plane):

        if not self.interesting_slice:
            return self.slice_id, plane

        image = subject[self.target_label_map_name]

        if 'interesting_slice_ids' not in image:
            image = FindInterestingSlice()(image)

        interesting_slice_ids = image['interesting_slice_ids']
        interesting_slice_counts = image['interesting_slice_counts']
        if plane.lower() == 'interesting':
            count = -1
            for check_plane in ("Axial", "Coronal", "Saggital"):
                new_count = self.get_slice_property(image, interesting_slice_counts, self.slice_id, check_plane)
                if new_count > count:
                    plane = check_plane
                    count = new_count
        else:
            plane = plane

        return self.get_slice_property(image, interesting_slice_ids, self.slice_id, plane), plane

    def get_slice_property(self, image, slice_property, slice_id, plane):
        _, W, H, D = image.data.shape
        dim = {'Axial': D, 'Coronal': H, 'Saggital': W}[plane]

        if slice_property[plane].shape[0] == 0:
            return dim // 2
        if slice_id >= slice_property[plane].shape[0]:
            return slice_property[plane][-1]
        return slice_property[plane][slice_id]

    def slice_and_make_grid(self, subjects, plane, image_name, label_value, impute_shape, pad_value=0):
        slices = []
        for subject in subjects:
            slice_id, plane = self.get_slice_id(subject, plane)
            if image_name in subject:
                slices.append(slice_volume(subject[image_name].data == label_value, 0, plane, slice_id).unsqueeze(0))
            else:
                slices.append(torch.zeros(impute_shape))

        return make_grid(slices, nrow=self.ncol, pad_value=pad_value, padding=1)[0].cpu()

    def __call__(self, subjects):
        if not self.split_subjects:
            return self.get_image(subjects)
        else:
            return {
                subject['name']: self.get_image([subject])
                for subject in subjects
            }

    def get_image(self, subjects):
        out_pred = self.prediction_label_map_name is not None
        out_target = self.target_label_map_name is not None

        if out_pred:
            label_values = subjects[0][self.prediction_label_map_name]['label_values']
        if out_target:
            label_values = subjects[0][self.target_label_map_name]['label_values']

        if self.plane.lower() == 'random':
            plane = ("Axial", "Coronal", "Saggital")[random.randint(0, 2)]
        else:
            plane = self.plane

        sample_subject = subjects[0]
        slice_id, plane = self.get_slice_id(sample_subject, plane)
        sample_slice = slice_volume(sample_subject[self.image_name].data, 0, plane, 0)
        impute_shape = sample_slice.shape

        img = self.slice_and_make_grid(subjects, plane, self.image_name, 0, impute_shape, pad_value=-1)
        if out_target:
            y = {
                label_name: self.slice_and_make_grid(
                    subjects, plane, self.target_label_map_name, label_value, impute_shape).bool()
                for label_name, label_value in label_values.items()
            }
        if out_pred:
            y_pred = {
                label_name: self.slice_and_make_grid(
                    subjects, plane, self.prediction_label_map_name, label_value, impute_shape).bool()
                for label_name, label_value in label_values.items()
            }

        H, W = img.shape
        fig = plt.figure(figsize=tuple(np.array((W, H)) * self.scale))
        plt.imshow(img, cmap="gray",)
        X_grid, Y_grid = np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H))
        options = dict(linewidths=self.line_width, alpha=1.)
        warnings.filterwarnings("ignore")
        cmap = [None, "r", "g", "b", "y", "c", "m"] \
               + list(get_cmap("Accent").colors) + list(get_cmap("Dark2").colors) \
               + list(get_cmap("Set1").colors) + list(get_cmap("Set2").colors) \
               + list(get_cmap("tab20").colors)
        contours = []

        if out_target:
            for label_name, label_id in label_values.items():
                contour = plt.contour(X_grid, Y_grid, y[label_name], levels=[0.5], colors=cmap[label_id:label_id+1],
                                      **options)
                contours.append(contour)
                if self.legend:
                    plt.legend(
                        [contour.legend_elements()[0][0] for contour in contours],
                        label_values.items(), ncol=3, bbox_to_anchor=(0.5, 0), loc='upper center', fancybox=True
                    )

        if out_pred:
            for label_name, label_id in label_values.items():
                plt.contour(X_grid, Y_grid, y_pred[label_name], levels=[0.95], linestyles="dashed",
                            colors=cmap[label_id:label_id+1], **options)

        warnings.resetwarnings()
        plt.tick_params(which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight", pad_inches=0.0, facecolor="black")
        buf.seek(0)
        pil_image = Image.open(buf)
        plt.close(fig)

        return pil_image
