""" 
Prints commands for parallel subjobs on a Compute Canada job array.

Update `params` and the print statement on a job specific basis

Each combination from the cartesian product of the items in `params` will be assigned
to a job array task.

To use with Compute Canada job script, pipe the output to a parallel command
ex `python research/dmri_hippo/generate_parallel_commands.py | parallel`

To bypass the environment variables for local testing, pass in the related args into main
"""

import itertools
import os
from pathlib import Path


params = {
    "aug_method": ["no_augmentation", "standard", "dwi_reconstruction", "combined"],
    "fold": range(0, 5)
}


def main(task_count, task_id, cpus_per_job, slurm_tmpdir):
    cart_product = list(itertools.product(*params.values()))

    configs = []
    for items in cart_product:
        configs.append(dict(zip(params.keys(), items)))

    # match task_id to congigs to run
    current_job_configs = []
    for i in range(len(configs)):
        if i % task_count == task_id:
            current_job_configs.append(configs[i])

    # determine number of cpus per config
    cpu_list = [0] * len(current_job_configs)
    for i in range(cpus_per_job):
        cpu_list[i % len(current_job_configs)] += 1


    for i, config in enumerate(current_job_configs):

        data_dir = Path(slurm_tmpdir) / f"{i}"
        data_dir.mkdir(parents=True, exist_ok=True)

        # make sure to leave space at the end of each line
        print(
            "python -m research.dmri_hippo.run augmentation_experiment "
            "~/projects/def-uofavis-ab/shared_data/Diffusion_MRI_cropped.tar "
            "~/scratch/Checkpoints/ "
            f"--work_path {data_dir} "
            f"--augmentation_mode {config['aug_method']} "
            "--max_training_time '0-8:0:0' "
            f"--num_cpu_threads {cpu_list[i]} "
            f"--fold {config['fold']} "
        )


if __name__ == "__main__":
    num_jobs = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    job_num = int(os.environ["SLURM_ARRAY_TASK_ID"])
    cpus_per_job = int(os.environ["SLURM_CPUS_PER_TASK"])
    slurm_tmpdir = os.environ["SLURM_TMPDIR"]

    main(num_jobs, job_num, cpus_per_job, slurm_tmpdir)