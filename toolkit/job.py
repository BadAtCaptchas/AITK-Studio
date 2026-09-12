"""Construct configured jobs and preserve failures while releasing their resources."""
from typing import Union, OrderedDict

from toolkit.config import get_config


def run_job_instance(job):
    """Run a constructed job and always attempt to release its resources.

    When both the job and cleanup fail, preserve the job failure as the
    actionable error and attach cleanup context using Python 3.11 exception notes.
    """
    primary_error = None
    try:
        job.run()
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            job.cleanup()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"Job cleanup also failed: {cleanup_error}")


def get_job(
        config_path: Union[str, dict, OrderedDict],
        name=None
):
    """Validate a configuration and construct only its selected job implementation."""
    config = get_config(config_path, name)
    if 'capability_version' in config:
        from toolkit.config_contract import config_contract_errors
        errors = config_contract_errors(config)
        if errors:
            raise ValueError(errors[0])
    if not config['job']:
        raise ValueError('config file is invalid. Missing "job" key')

    job = config['job']
    if job == 'extract':
        from jobs import ExtractJob
        return ExtractJob(config)
    if job == 'train':
        from jobs import TrainJob
        return TrainJob(config)
    if job == 'mod':
        from jobs import ModJob
        return ModJob(config)
    if job == 'generate':
        from jobs import GenerateJob
        return GenerateJob(config)
    if job == 'extension':
        from jobs import ExtensionJob
        return ExtensionJob(config)

    raise ValueError(f'Unknown job type {job}')


def run_job(
        config: Union[str, dict, OrderedDict],
        name=None
):
    """Construct and execute a configured job, including resource cleanup."""
    job = get_job(config, name)
    run_job_instance(job)
