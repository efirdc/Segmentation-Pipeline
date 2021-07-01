import time
import os
import signal
import threading
from typing import Sequence, Callable, Union
import copy
import warnings

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torchio as tio
from torchio.transforms.preprocessing.label.label_transform import LabelTransform
import wandb

from evaluators import *
from data_processing import *
from transforms import *
from utils import Timer, dict_to_device, filter_transform, to_wandb


EXIT = threading.Event()
EXIT.clear()


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)
if os.name != 'nt':
    signal.signal(signal.SIGUSR2, _clean_exit_handler)


class ScheduledEvaluation:
    def __init__(
            self,
            evaluator: Evaluator,
            log_name: str,
            cohorts: Sequence[str] = None,
            subjects: Sequence[str] = None,
            interval: int = None,
    ):
        assert not (cohorts and subjects), "One of cohorts or subjects may be provided, but not both."
        self.evaluator = evaluator
        self.log_name = log_name
        self.cohorts = cohorts
        self.subjects = subjects
        self.interval = interval


class SegmentationTrainer:
    def __init__(
            self,
            save_folder: str,
            training_batch_size: int,
            save_rate: int,
            scoring_interval: int,
            scoring_function: Callable,
            one_time_evaluators: Sequence[ScheduledEvaluation],
            training_evaluators: Sequence[ScheduledEvaluation],
            validation_evaluators: Sequence[ScheduledEvaluation],
            max_iterations_with_no_improvement: int
    ):
        self.save_folder = save_folder
        self.training_batch_size = training_batch_size
        self.save_rate = save_rate
        self.scoring_interval = scoring_interval
        self.scoring_function = scoring_function
        self.one_time_evaluators = one_time_evaluators
        self.training_evaluators = training_evaluators
        self.validation_evaluators = validation_evaluators
        self.max_iterations_with_no_improvement = max_iterations_with_no_improvement

        self.max_score = -1
        self.max_score_iteration = -1

    def train(self, context, iterations, stop_time=None, wandb_logging=True, preload_training_data=False,
              preload_validation_data=False, validation_batch_size=16):

        # Initialize directories for saving models and images
        save_folder = f'{self.save_folder}/{context.name}/'
        image_folder = save_folder + "images/"
        checkpoints_folder = save_folder + "checkpoints/"
        best_checkpoints_folder = save_folder + "best_checkpoints/"
        for folder in (image_folder, checkpoints_folder, best_checkpoints_folder):
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Function for saving the context
        def save_context(path):
            context.save(f"{path}iter{context.iteration:08}.pt")

        # Get the training_dataset
        training_dataset = context.dataset.get_cohort_dataset("training")
        training_dataset.collate_attributes += ['X', 'y']
        if preload_training_data:
            t = time.time()
            print("Preloading training data...")
            training_dataset.preload_subjects()
            print(f"Done. Took {round(time.time() - t, 2)}s")

        # Infer the validation_dataset from scheduled evaluations
        validation_filter = self.get_filter_from_scheduled_evaluations(context.dataset, self.validation_evaluators)
        validation_dataset = context.dataset.get_cohort_dataset(validation_filter)
        validation_dataset.collate_attributes += ['X']
        if preload_validation_data:
            t = time.time()
            print("Preloading validation data...")
            validation_dataset.preload_and_transform_subjects()
            print(f"Done. Took {round(time.time() - t, 2)}s")

        # Make dataloaders for training and validation datasets
        training_dataloader = DataLoader(dataset=training_dataset, batch_size=self.training_batch_size,
                                         sampler=RandomSampler(training_dataset),
                                         collate_fn=training_dataset.collate)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_batch_size,
                                           sampler=SequentialSampler(validation_dataset),
                                           collate_fn=validation_dataset.collate)

        # Make an iterator for the training dataset
        def get_data_sampler(loader):
            while True:
                for batch in loader:
                    yield batch
        training_data_sampler = get_data_sampler(training_dataloader)

        # Grab an instance of the target label from the training dataset.
        # Some subjects in the validation set might not have it, so this is used to fill in attributes
        y_sample = training_dataset[0]['y']
        y_sample.set_data(torch.ones(1, 1, 1, 1))

        # Training loop
        timer = Timer(context.device)
        for _ in range(iterations):
            timer.start()

            batch = next(training_data_sampler)
            batch['X']['data'] = batch['X']['data'].to(context.device)
            batch['y']['data'] = batch['y']['data'].to(context.device)
            timer.stamp("data_loading")

            context.model.train()
            self.seg_predict(context.model, batch, y_sample)
            timer.stamp("model_forward")

            loss_dict = context.criterion(batch['y_pred']['data'], batch['y']['data'])
            timer.stamp("loss_function")

            context.optimizer.zero_grad()
            loss_dict['loss'].backward()
            context.optimizer.step()
            context.model.eval()
            timer.stamp("model_backward")

            training_evaluations = {}
            for scheduled in self.training_evaluators:
                training_evaluations[scheduled.log_name] = scheduled.evaluator(batch['subjects'])
                timer.stamp(f"evaluation.{scheduled.log_name}")

            # Determine which scheduled evaluators should run this iteration
            validation_evaluations = {}
            validation_evaluators = [
                scheduled for scheduled in self.validation_evaluators
                if context.iteration % scheduled.interval == 0
            ]

            # Run evaluations if at least one is scheduled
            if len(validation_evaluators) > 0:

                # Run every subject that is scheduled to be evaluated through the model
                validation_filter = self.get_filter_from_scheduled_evaluations(validation_dataset, validation_evaluators)
                validation_dataset.set_cohort(validation_filter)
                validation_subjects = []
                for batch in validation_dataloader:
                    batch['X']['data'] = batch['X']['data'].to(context.device)
                    with torch.no_grad():
                        self.seg_predict(context.model, batch, y_sample)
                    validation_subjects += batch['subjects']
                validation_subjects_map = {subject['name']: subject for subject in validation_subjects}
                timer.stamp('model_forward_evaluation')

                # Run the evaluations
                for scheduled in validation_evaluators:

                    # Handle evaluations that run on multiple cohorts
                    # The output of each is logged as {log_name: {cohort_name: evaluation_output}}
                    if scheduled.cohorts is not None:
                        validation_evaluations[scheduled.log_name] = cohort_evaluations = {}
                        for cohort_name in scheduled.cohorts:
                            subject_filter = validation_dataset.cohorts[cohort_name]
                            filtered_subjects = subject_filter(validation_subjects)
                            cohort_evaluations[cohort_name] = scheduled.evaluator(filtered_subjects)
                            timer.stamp(f"evaluation.{scheduled.log_name}.{cohort_name}")

                    # Handle scheduled evaluations that run on a pre-defined list of subjects (given by name)
                    # The output of each is logged as {log_name: evaluation_output}
                    elif scheduled.subjects is not None:
                        filtered_subjects = [validation_subjects_map[subject_name]
                                             for subject_name in scheduled.subjects]
                        validation_evaluations[scheduled.log_name] = scheduled.evaluator(filtered_subjects)
                        timer.stamp(f"evaluation.{scheduled.log_name}")

            log_dict = {**loss_dict, **training_evaluations, **validation_evaluations}

            if context.iteration % self.save_rate == 0:
                save_context(checkpoints_folder)
                timer.stamp("save_checkpoint")

            if context.iteration % self.scoring_interval == 0:
                new_score = self.scoring_function(log_dict)
                log_dict['model_score'] = new_score

                if new_score > self.max_score:
                    self.max_score = new_score
                    self.max_score_iteration = context.iteration
                    save_context(best_checkpoints_folder)
                    timer.stamp("save_best_checkpoint")

            log_dict['timer'] = timer.timestamps

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wandb.log(to_wandb(log_dict))

            context.iteration += 1

            iterations_with_no_improvement = context.iteration - self.max_score_iteration
            if iterations_with_no_improvement > self.max_iterations_with_no_improvement:
                print(f"Training stopped on iteration {context.iteration} due to not improving for "
                      f"{iterations_with_no_improvement} iterations.")
                break

            if EXIT.is_set() or (stop_time is not None and time.time() > stop_time):
                if EXIT.is_set():
                    print("Training stopped early due to manual exit signal.")
                else:
                    print("Training time expired.")
                break

        print("Saving context...")
        save_context(checkpoints_folder)

    def get_filter_from_scheduled_evaluations(
            self,
            dataset: SubjectFolder,
            scheduled_evaluations: Sequence[ScheduledEvaluation]
    ):
        filters = []
        for scheduled_evaluation in scheduled_evaluations:
            if scheduled_evaluation.cohorts is not None:
                cohort_names = scheduled_evaluation.cohorts
                filters += [dataset.cohorts[cohort_name] for cohort_name in cohort_names]
            elif scheduled_evaluation.subjects is not None:
                subject_names = scheduled_evaluation.subjects
                filters.append(RequireAttributes({'name': subject_names}))
        subject_filter = AnyFilter(filters)
        return subject_filter

    def seg_predict(self, model, batch, sample_y):
        batch['y_pred'] = {'data': model(batch['X']['data'])}

        for i in range(len(batch['subjects'])):
            subject = batch['subjects'][i]
            y_pred = copy.deepcopy(sample_y)
            y_pred.set_data(batch['y_pred']['data'][i].detach().cpu())
            subject['y_pred'] = y_pred

            transform = subject.get_composed_history()
            label_transform_types = [LabelTransform, CopyProperty, RenameProperty, ConcatenateImages]
            label_transform = filter_transform(transform, include_types=label_transform_types)
            inverse_label_transform = label_transform.inverse(warn=False)

            evaluation_transform = tio.Compose([
                inverse_label_transform,
                CustomSequentialLabels(),
                filter_transform(inverse_label_transform, exclude_types=[CustomRemapLabels]).inverse(warn=False)
            ])

            pred_subject = tio.Subject({'y': y_pred})
            subject['y_pred_eval'] = evaluation_transform(pred_subject)['y']

            if 'y' in subject:
                target_subject = tio.Subject({'y': subject['y']})
                subject['y_eval'] = evaluation_transform(target_subject)['y']