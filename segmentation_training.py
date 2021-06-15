from typing import Sequence
import time
import warnings

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchio as tio
import os
import signal
import threading

from torchvision.utils import make_grid

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import io

import wandb

from evaluation import dice_validation
from utils import CudaTimer
import copy


EXIT = threading.Event()
EXIT.clear()


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)
if os.name != 'nt':
    signal.signal(signal.SIGUSR2, _clean_exit_handler)


def seg_predict(model, batch, inverse_label_transform):
    batch['y_pred'] = model(batch)

    for i in range(len(batch['subjects'])):
        subject = batch['subjects'][i]
        y_pred_subject = batch['y_pred'][i]

        subject['y_pred'] = copy.deepcopy(subject['y'])
        subject['y_pred'].set_data(y_pred_subject)
        subject['y_inverse'] = inverse_label_transform(subject['y'])
        subject['y_pred_inverse'] = inverse_label_transform(subject['y_pred'])


class SegmentationTrainer:
    def __init__(
            self,
            save_folder: str,
            save_rate,
            training_evaluators,
            validation_evaluator_schedule,
    ):
        self.save_folder = save_folder
        self.save_rate = save_rate
        self.training_evaluators = training_evaluators
        self.validation_evaluator_schedule = validation_evaluator_schedule

    def train(self, context, iterations, stop_time=None, wandb_logging=False,
              preload_training_dataset=False, preload_validation_datasets=False):

        save_folder = f'{self.save_folder}/{context.name}/'
        image_folder = save_folder + "images/"
        for folder in (save_folder, image_folder):
            if not os.path.exists(folder):
                os.makedirs(folder)

        def save_context():
            context.save(f"{save_folder}/iter{context.iteration:08}.pt")

        def sample_data(loader):
            while True:
                for batch in loader:
                    yield batch

        if preload_training_dataset:
            print("Preloading training data.")
            context.dataset.preload_subjects()

        if preload_validation_datasets:
            print("Preloading validation data.")
            for val_dataset in self.val_datasets:
                if val_dataset["preload"]:
                    val_dataset.update(val_dataset['dataset'].load_and_collate_all_subjects())
            print("Preload finished.")

        loader = sample_data(context.dataloader)

        timer = CudaTimer()
        for _ in range(iterations):
            timer.start()

            logging_dict = {}

            batch = next(loader)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(context.device)
            timer.stamp("data_loading")

            context.model.train()
            seg_predict(context.model, batch, context.dataset.inverse_label_transform)
            timer.stamp("model_forward")

            loss_dict = context.criterion(batch)
            timer.stamp("loss_function")

            context.optimizer.zero_grad()
            loss_dict['loss'].backward()
            context.optimizer.step()
            context.model.eval()
            timer.stamp("model_backward")

            training_evaluations = {}
            for evaluator in self.training_evaluators:
                training_evaluations[evaluator.name] = evaluator.evaluate(batch['subjects'])
                timer.stamp(f"training_evaluation.{evaluator.name}")

            # A cache in order to avoid predicting validation subjects more than once per iteration
            # Maps subject_name -> (X, y, y_pred)
            cache = {}
            def predict(subject=None, subjects=None, X=None, y=None):
                if X is None or y is None:
                    if subject:
                        if subject.name in cache:
                            return cache[subject.name]
                        subjects = [subject]
                    X, y = self.collate_subjects(subjects)
                X = X.to(context.device)
                y = y.to(context.device)
                with torch.no_grad():
                    y_pred = context.model(X)
                y_pred = inverse_label_transform(y_pred.argmax(dim=1))
                y = inverse_label_transform(y)
                out = (X, y, y_pred)
                for i in range(len(subjects)):
                    cache[subjects[i].name] = (X[i:i+1], y[i:i+1], y_pred[i:i+1])
                return out

            val_dice_results = {}
            for val_dataset in self.val_datasets:
                if context.iteration % val_dataset["interval"] != 0:
                    continue
                if val_dataset["preload"]:
                    _, y, y_pred = predict(
                        subjects=val_dataset["dataset"].subjects, X=val_dataset["X"], y=val_dataset["y"]
                    )
                else:
                    subjects = [val_dataset["dataset"][i] for i in range(len(val_dataset["dataset"]))]
                    _, y, y_pred = predict(subjects=subjects)
                val_dice_results.update(dice_validation(y_pred, y, label_names_inverse, val_dataset["log_prefix"]))
            timer.stamp("Time: Validation Dice")


            wandb.log({**loss_dict, **train_dice_results, **val_dice_results, **timer.timestamps})

            context.iteration += 1

            if EXIT.is_set() or (stop_time is not None and time.time() > stop_time):
                if EXIT.is_set():
                    print("Training stopped early due to manual exit signal.")
                else:
                    print("Training time expried.")
                break

        print("Saving context...")
        save_context()