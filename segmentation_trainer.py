import time
import os
import signal
import threading
from typing import Sequence, Callable, Union, Tuple
import copy

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torchio as tio
from torchio.transforms.preprocessing.label.label_transform import LabelTransform
from torchio.data.sampler.sampler import PatchSampler

from evaluators import *
from data_processing import *
from loggers import *
from transforms import *
from utils import Timer, filter_transform
from segmentation import *


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
            interval: int = 1,
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
            training_batch_size: int,
            save_rate: int,
            scoring_interval: int,
            scoring_function: Callable,
            one_time_evaluators: Sequence[ScheduledEvaluation],
            training_evaluators: Sequence[ScheduledEvaluation],
            validation_evaluators: Sequence[ScheduledEvaluation],
            max_iterations_with_no_improvement: int,
            enable_patch_mode: bool = False,
            patch_size: Union[int, Tuple[int, int, int]] = None,
            training_patch_sampler: PatchSampler = None,
            training_patches_per_volume: int = None,
            validation_patch_overlap: Union[int, Tuple[int, int, int]] = None,
            validation_padding_mode: Union[str, float, None] = None,
            validation_overlap_mode: str = 'average',
    ):
        self.training_batch_size = training_batch_size
        self.save_rate = save_rate
        self.scoring_interval = scoring_interval
        self.scoring_function = scoring_function
        self.one_time_evaluators = one_time_evaluators
        self.training_evaluators = training_evaluators
        self.validation_evaluators = validation_evaluators
        self.max_iterations_with_no_improvement = max_iterations_with_no_improvement
        self.enable_patch_mode = enable_patch_mode
        self.patch_size = patch_size
        self.training_patch_sampler = training_patch_sampler
        self.training_patches_per_volume = training_patches_per_volume
        self.validation_patch_overlap = validation_patch_overlap
        self.validation_padding_mode = validation_padding_mode
        self.validation_overlap_mode = validation_overlap_mode

        self.iteration = 0
        self.max_score = float('-inf')
        self.max_score_iteration = -1

    def state_dict(self):
        return {
            'iteration': self.iteration,
            'max_score': self.max_score,
            'max_score_iteration': self.max_score_iteration,
        }

    def load_state_dict(self, state):
        self.iteration = state['iteration']
        self.max_score = state['max_score']
        self.max_score_iteration = state['max_score_iteration']

    def train(
            self,
            context,
            iterations,
            stop_time=None,
            preload_training_data=False,
            preload_validation_data=False,
            num_workers=0,
            validation_batch_size=16,
            validation_patch_batch_size=32,
            patch_queue_length=100,
            logger=NonLogger(),
            **kwargs
    ):
        print("Initializing logger.")
        logger.setup(context)

        # Get the training_dataset
        training_dataset = context.dataset.get_cohort_dataset("training")
        if preload_training_data:
            t = time.time()
            print("Preloading training data...")
            training_dataset.preload_subjects()
            print(f"Done. Took {round(time.time() - t, 2)}s")

        # Infer the validation_dataset from scheduled evaluations
        validation_filter = self.get_filter_from_scheduled_evaluations(context.dataset, self.validation_evaluators)
        validation_dataset = context.dataset.get_cohort_dataset(validation_filter)
        if preload_validation_data:
            t = time.time()
            print("Preloading validation data...")
            validation_dataset.preload_and_transform_subjects()
            print(f"Done. Took {round(time.time() - t, 2)}s")

        # Make dataloader for training dataset
        if not self.enable_patch_mode:
            training_dataloader = DataLoader(dataset=training_dataset,
                                             batch_size=self.training_batch_size,
                                             sampler=RandomSampler(training_dataset),
                                             collate_fn=dont_collate,
                                             num_workers=num_workers)
        else:
            queue = tio.Queue(training_dataset,
                              max_length=patch_queue_length,
                              samples_per_volume=self.training_patches_per_volume,
                              sampler=self.training_patch_sampler,
                              num_workers=num_workers)
            training_dataloader = DataLoader(dataset=queue,
                                             batch_size=self.training_batch_size,
                                             collate_fn=dont_collate)

        # Make an iterator for the training dataset
        def get_data_iterator(loader):
            while True:
                for batch in loader:
                    yield batch
        training_data_iterator = get_data_iterator(training_dataloader)

        # Grab an instance of the target label from the training dataset.
        # Some subjects in the validation set might not have it, so this is used to fill in attributes
        y_sample = training_dataset[0]['y']
        y_sample.set_data(torch.ones(1, 1, 1, 1))

        # Training loop
        timer = Timer(context.device)
        for _ in range(iterations):
            timer.start()

            subjects = next(training_data_iterator)
            batch = collate_subjects(subjects, image_names=['X', 'y'], device=context.device)
            timer.stamp("data_loading")

            context.model.train()
            seg_predict(context.model, batch, subjects, y_sample)
            timer.stamp("model_forward")

            loss_dict = context.criterion(batch['y_pred'], batch['y'])
            timer.stamp("loss_function")

            context.optimizer.zero_grad()
            loss_dict['loss'].backward()
            context.optimizer.step()
            context.model.eval()
            timer.stamp("model_backward")

            training_evaluations = {}
            training_evaluators = [
                scheduled for scheduled in self.training_evaluators
                if self.iteration % scheduled.interval == 0
            ]
            if len(training_evaluators) > 0:
                add_evaluation_labels(subjects)
            for scheduled in training_evaluators:
                training_evaluations[scheduled.log_name] = scheduled.evaluator(subjects)
                timer.stamp(f"evaluation.{scheduled.log_name}")

            # Determine which scheduled evaluators should run this iteration
            validation_evaluations = {}
            validation_evaluators = [
                scheduled for scheduled in self.validation_evaluators
                if self.iteration % scheduled.interval == 0
            ]

            # Run evaluations if at least one is scheduled
            if len(validation_evaluators) > 0:

                # Run every subject that is scheduled to be evaluated through the model
                validation_filter = self.get_filter_from_scheduled_evaluations(context.dataset,
                                                                               validation_evaluators)
                validation_dataset.set_cohort(validation_filter)
                validation_dataloader = DataLoader(dataset=validation_dataset,
                                                   batch_size=validation_batch_size,
                                                   sampler=SequentialSampler(validation_dataset),
                                                   collate_fn=dont_collate,
                                                   num_workers=num_workers)
                validation_subjects = []
                with torch.no_grad():
                    for subjects in validation_dataloader:
                        if not self.enable_patch_mode:
                            batch = collate_subjects(subjects, image_names=['X'], device=context.device)
                            seg_predict(context.model, batch, subjects, y_sample)
                        else:
                            patch_predict(context.model, subjects, y_sample,
                                          patch_batch_size=validation_patch_batch_size,
                                          patch_size=self.patch_size,
                                          patch_overlap=self.validation_patch_overlap,
                                          padding_mode=self.validation_padding_mode,
                                          overlap_mode=self.validation_overlap_mode)
                        add_evaluation_labels(subjects)
                        validation_subjects += subjects

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

            if self.iteration % self.save_rate == 0:
                logger.save_context(context, "checkpoints/", self.iteration)
                timer.stamp("save_checkpoint")

            if self.iteration % self.scoring_interval == 0:
                new_score = self.scoring_function(log_dict)
                log_dict['model_score'] = new_score

                if new_score > self.max_score:
                    self.max_score = new_score
                    self.max_score_iteration = self.iteration
                    logger.save_context(context, "best_checkpoints/", self.iteration)
                    timer.stamp("save_best_checkpoint")

            log_dict['timer'] = timer.timestamps

            logger.log(log_dict)

            iterations_with_no_improvement = self.iteration - self.max_score_iteration
            if iterations_with_no_improvement > self.max_iterations_with_no_improvement:
                print(f"Training stopped on iteration {self.iteration} due to not improving for "
                      f"{iterations_with_no_improvement} iterations.")
                break

            if EXIT.is_set() or (stop_time is not None and time.time() > stop_time):
                if EXIT.is_set():
                    print("Training stopped early due to manual exit signal.")
                else:
                    print("Training time expired.")
                break

            self.iteration += 1

        print("Saving context...")
        logger.save_context(context, "checkpoints/", self.iteration)

    def get_filter_from_scheduled_evaluations(
            self,
            dataset: SubjectFolder,
            scheduled_evaluations: Sequence[ScheduledEvaluation],
    ):
        filters = []
        for scheduled_evaluation in scheduled_evaluations:
            if scheduled_evaluation.cohorts is not None:
                cohort_names = scheduled_evaluation.cohorts

                filters += [
                    dataset.cohorts[cohort_name] for cohort_name in cohort_names
                ]

            elif scheduled_evaluation.subjects is not None:
                subject_names = scheduled_evaluation.subjects
                filters.append(RequireAttributes({'name': subject_names}))
        subject_filter = AnyFilter(filters)
        return subject_filter
