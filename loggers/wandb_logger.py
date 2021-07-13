import os
import warnings
from typing import Optional

import wandb
from random_words import RandomWords
import pandas as pd
from PIL import Image

from .logger import Logger


def to_wandb(elem):
    if isinstance(elem, dict):
        return {
            key: to_wandb(val)
            for key, val in elem.items()
        }
    elif isinstance(elem, pd.DataFrame):
        return wandb.Table(dataframe=elem)
    elif isinstance(elem, Image.Image):
        return wandb.Image(elem)
    return elem


class WandbLogger(Logger):
    """ Logger implementation for Weights and Biases logging service.

    Args:
        project_name: Project name in weights and biases
        logging_dir: Root directory for logging experiments. The current run will be logged in:
            logging_dir/project_name/wandb_run_name/
    """
    def __init__(
            self,
            project_name: str,
            logging_dir: str,
            group_name: Optional[str] = None
    ):
        self.project_name = project_name
        self.logging_dir = logging_dir
        self.group_name = group_name

    def setup(self, context):
        wandb_params = {
            'project': self.project_name,
        }

        if self.group_name is not None:
            wandb_params['group'] = self.group_name

        resuming_previous_run = 'wandb_id' in context.metadata
        if not resuming_previous_run:
            wandb_params['id'] = context.metadata["wandb_id"] = wandb.util.generate_id()
            rw = RandomWords()
            context.name = f'{context.name}-{rw.random_word()}-{rw.random_word()}-{context.metadata["wandb_id"]}'
            wandb_params['name'] = context.name
            wandb_params['config'] = context.get_config()
        else:
            wandb_params['id'] = context.metadata["wandb_id"]
            wandb_params['resume'] = 'allow'

        # Initialize directories for saving data
        if self.group_name is None:
            self.save_folder = os.path.join(self.logging_dir, self.project_name, context.name)
        else:
            self.save_folder = os.path.join(self.logging_dir, self.project_name, self.group_name, context.name)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        wandb_params['dir'] = self.save_folder
        wandb.init(**wandb_params)

        # Save code on first iteration
        if not resuming_previous_run:
            for file_path in context.file_paths:
                wandb.save(file_path)

        print(str(context))

    def save_context(self, context, sub_folder, iteration):
        sub_dir = os.path.join(self.save_folder, sub_folder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        file_path = os.path.join(sub_dir, f"{context.name}-iter{iteration:08}.pt")
        context.save(file_path)
        wandb.save(file_path)

    def log(self, log_dict):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wandb.log(to_wandb(log_dict))
