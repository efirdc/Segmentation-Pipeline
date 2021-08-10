import os
import warnings
import shutil
import glob
from typing import Optional, List, Union
from six import string_types

import wandb
from wandb.sdk.lib import telemetry
from random_words import RandomWords
import pandas as pd
from PIL import Image

from .logger import Logger
from ..utils import flatten_nested_dict
from ..evaluators import LabeledTensor


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
    elif isinstance(elem, LabeledTensor):
        return elem.to_dict()
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
            'settings': wandb.Settings(symlink=False),
        }

        if self.group_name is not None:
            wandb_params['group'] = self.group_name

        resuming_previous_run = 'wandb_id' in context.metadata
        if not resuming_previous_run:
            wandb_params['id'] = context.metadata["wandb_id"] = wandb.util.generate_id()

            rw = RandomWords()
            context.name = f'{context.name}-{rw.random_word()}-{rw.random_word()}-{context.metadata["wandb_id"]}'
            wandb_params['name'] = context.name

            config = context.get_config()
            config = flatten_nested_dict(config)
            wandb_params['config'] = config
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
                WandbLogger.wandb_save(file_path)

        print(str(context))

    def save_context(self, context, sub_folder, iteration):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning)
            sub_dir = os.path.join(self.save_folder, sub_folder)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            file_path = os.path.join(sub_dir, f"{context.name}-iter{iteration:08}.pt")
            context.save(file_path)
            WandbLogger.wandb_save(file_path)

    def log(self, log_dict):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning)
            wandb.log(to_wandb(log_dict))

    @staticmethod
    def wandb_save(
            glob_str: Optional[str] = None,
            base_path: Optional[str] = None,
            policy: str = "live",
    ) -> Union[bool, List[str]]:
        """
        NOTE: This reimplements wandb.save, but copies files instead of symlinking.
        The symlinks have caused many issues on Windows and google colab.

        ORIGINAL DOCS:
        Ensure all files matching `glob_str` are synced to wandb with the policy specified.

        Arguments:
            glob_str: (string) a relative or absolute path to a unix glob or regular
                path.  If this isn't specified the method is a noop.
            base_path: (string) the base path to run the glob relative to
            policy: (string) on of `live`, `now`, or `end`
                - live: upload the file as it changes, overwriting the previous version
                - now: upload the file once now
                - end: only upload file when the run ends
        """
        if glob_str is None:
            # noop for historical reasons, run.save() may be called in legacy code
            wandb.termwarn(
                (
                    "Calling run.save without any arguments is deprecated."
                    "Changes to attributes are automatically persisted."
                )
            )
            return True
        if policy not in ("live", "end", "now"):
            raise ValueError(
                'Only "live" "end" and "now" policies are currently supported.'
            )
        if isinstance(glob_str, bytes):
            glob_str = glob_str.decode("utf-8")
        if not isinstance(glob_str, string_types):
            raise ValueError("Must call wandb.save(glob_str) with glob_str a str")

        if base_path is None:
            if os.path.isabs(glob_str):
                base_path = os.path.dirname(glob_str)
                wandb.termwarn(
                    (
                        "Saving files without folders. If you want to preserve "
                        "sub directories pass base_path to wandb.save, i.e. "
                        'wandb.save("/mnt/folder/file.h5", base_path="/mnt")'
                    )
                )
            else:
                base_path = ""
        wandb_glob_str = os.path.relpath(glob_str, base_path)
        if ".." + os.sep in wandb_glob_str:
            raise ValueError("globs can't walk above base_path")

        with telemetry.context(run=wandb.run) as tel:
            tel.feature.save = True

        if glob_str.startswith("gs://") or glob_str.startswith("s3://"):
            wandb.termlog(
                "%s is a cloud storage url, can't save file to wandb." % glob_str
            )
            return []
        files = glob.glob(os.path.join(wandb.run.dir, wandb_glob_str))
        warn = False
        if len(files) == 0 and "*" in wandb_glob_str:
            warn = True
        for path in glob.glob(glob_str):
            file_name = os.path.relpath(path, base_path)
            abs_path = os.path.abspath(path)
            wandb_path = os.path.join(wandb.run.dir, file_name)
            wandb.util.mkdir_exists_ok(os.path.dirname(wandb_path))
            # We overwrite symlinks because namespaces can change in Tensorboard
            if os.path.islink(wandb_path) and abs_path != os.readlink(wandb_path):
                os.remove(wandb_path)
                shutil.copy(abs_path, wandb.run.dir)  # os.symlink(abs_path, wandb_path)
            elif not os.path.exists(wandb_path):
                shutil.copy(abs_path, wandb.run.dir)  # os.symlink(abs_path, wandb_path)
            files.append(wandb_path)
        if warn:
            file_str = "%i file" % len(files)
            if len(files) > 1:
                file_str += "s"
            wandb.termwarn(
                (
                    "Symlinked %s into the W&B run directory, "
                    "call wandb.save again to sync new files."
                )
                % file_str
            )
        files_dict = dict(files=[(wandb_glob_str, policy)])
        if wandb.run._backend:
            wandb.run._backend.interface.publish_files(files_dict)
        return files