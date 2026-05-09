import os
import sys
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
seed = None
if "SEED" in os.environ:
    try:
        seed = int(os.environ["SEED"])
    except ValueError:
        print(f"Invalid SEED value: {os.environ['SEED']}. SEED must be an integer.")

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

from toolkit.hf_download_progress import install_hf_download_progress

install_hf_download_progress()

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# set torch to trace mode
import torch
from toolkit.cuda_compat import check_blackwell_cuda_compatibility

check_blackwell_cuda_compatibility(torch)
    
# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    torch.autograd.set_detect_anomaly(True)

if seed is not None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file
from toolkit.exceptions import JobStopRequested

accelerator = get_accelerator()


def print_end_message(jobs_completed, jobs_failed, jobs_stopped=0):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    stopped_string = f"{jobs_stopped} stopped job{'' if jobs_stopped == 1 else 's'}" if jobs_stopped > 0 else ""

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(stopped_string) > 0:
        print_acc(f" - {stopped_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def get_first_process(job):
    if job is None:
        return None
    process_list = getattr(job, "process", None)
    if not process_list:
        return None
    return process_list[0]


def mark_process_stopping(process, status=None, info=None):
    if process is None:
        return
    if hasattr(process, "is_stopping"):
        process.is_stopping = True
    if status is not None and hasattr(process, "update_status"):
        try:
            process.update_status(status, info)
        except Exception as e:
            print_acc(f"Error updating stop status: {e}")


def run_process_error_handler(process, e):
    if process is None:
        return
    try:
        process.on_error(e)
    except Exception as e2:
        print_acc(f"Error running on_error: {e2}")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    args = parser.parse_args()
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0
    jobs_stopped = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        job = None
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except JobStopRequested as e:
            jobs_stopped += 1
            process = get_first_process(job)
            mark_process_stopping(process)
            run_process_error_handler(process, e)
            print_end_message(jobs_completed, jobs_failed, jobs_stopped)
            return
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            run_process_error_handler(get_first_process(job), e)
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed, jobs_stopped)
                raise e
        except KeyboardInterrupt as e:
            jobs_stopped += 1
            process = get_first_process(job)
            mark_process_stopping(process, "stopped", "Job stopped")
            run_process_error_handler(process, e)
            print_end_message(jobs_completed, jobs_failed, jobs_stopped)
            return


if __name__ == '__main__':
    main()
