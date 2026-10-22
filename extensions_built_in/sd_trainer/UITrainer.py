from collections import OrderedDict
import os
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
import threading
import time
import signal
from toolkit.exceptions import JobStopRequested
from toolkit.ui_database import UIJobStore
from toolkit.ui_progress import UIReportingMixin

class UITrainer(UIReportingMixin, SDTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(UITrainer, self).__init__(process_id, job, config, **kwargs)
        self.sqlite_db_path = self.config.get("sqlite_db_path", "./aitk_db.db")
        self.job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = self.job_id.strip() if self.job_id is not None else None
        self.ui_job_store = UIJobStore(self.job_id, self.sqlite_db_path)
        if not self.ui_job_store.available:
            raise Exception("UI database not available for trainer")
        print(f"Using {self.ui_job_store.description}")
        print(f"Job ID: \"{self.job_id}\"")
        if self.job_id is None:
            raise Exception("AITK_JOB_ID not set")
        self.is_stopping = False
        self._init_ui_reporting()
        self.update_status("running", "Starting")
        self._stop_watcher_started = False
        # self.start_stop_watcher(interval_sec=2.0)

    def start_stop_watcher(self, interval_sec: float = 5.0):
        """
        Start a daemon thread that periodically checks should_stop()
        and terminates the process immediately when triggered.
        """
        if getattr(self, "_stop_watcher_started", False):
            return
        self._stop_watcher_started = True
        t = threading.Thread(
            target=self._stop_watcher_thread, args=(interval_sec,), daemon=True
        )
        t.start()

    def _stop_watcher_thread(self, interval_sec: float):
        while True:
            try:
                if self.should_stop():
                    # Mark and update status (non-blocking; uses existing infra)
                    self.is_stopping = True
                    self.update_status("stopped", "Job stopped (remote)")
                    self._close_ui_reporting()
                    print("")
                    print("****************************************************")
                    print("    Stop signal received; terminating process.      ")
                    print("****************************************************")
                    os.kill(os.getpid(), signal.SIGINT)
                time.sleep(interval_sec)
            except Exception:
                time.sleep(interval_sec)


    def should_stop(self):
        return self.ui_job_store.should_stop()

    def should_return_to_queue(self):
        return self.ui_job_store.should_return_to_queue()

    def maybe_stop(self):
        if self.should_stop():
            self.update_status("stopped", "Job stopped")
            self.is_stopping = True
            raise JobStopRequested("Job stopped")
        if self.should_return_to_queue():
            self.update_status("queued", "Job queued")
            self.is_stopping = True
            raise JobStopRequested("Job returning to queue", return_to_queue=True)


    def on_error(self, e: Exception):
        super(UITrainer, self).on_error(e)
        self.update_db_key("step", self.last_save_step)
        if self.accelerator.is_main_process and not self.is_stopping:
            self.update_status("error", str(e))
        self.flush_ui_reporting()
        self._close_ui_reporting()
        self.ui_job_store.close()

    def handle_timing_print_hook(self, timing_dict):
        if "train_loop" not in timing_dict:
            print("train_loop not found in timing_dict", timing_dict)
            return
        seconds_per_iter = timing_dict["train_loop"]
        # determine iter/sec or sec/iter
        if seconds_per_iter < 1:
            iters_per_sec = 1 / seconds_per_iter
            self.update_db_key("speed_string", f"{iters_per_sec:.2f} iter/sec")
        else:
            self.update_db_key(
                "speed_string", f"{seconds_per_iter:.2f} sec/iter")

    def done_hook(self):
        super(UITrainer, self).done_hook()
        self.update_status("completed", "Training completed")
        # Wait for all async operations to finish before shutting down
        self.flush_ui_reporting()
        self._close_ui_reporting()
        self.ui_job_store.close()

    def end_step_hook(self):
        super(UITrainer, self).end_step_hook()
        self.update_step()
        self.maybe_stop()

    def hook_before_model_load(self):
        super().hook_before_model_load()
        self.maybe_stop()
        self.update_status("running", "Loading model")

    def before_dataset_load(self):
        super().before_dataset_load()
        self.maybe_stop()
        self.update_status("running", "Loading dataset")

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self.maybe_stop()
        self.update_step()
        self.update_status("running", "Training")
        self.timer.add_after_print_hook(self.handle_timing_print_hook)

    def status_update_hook_func(self, string):
        self.update_status("running", string)

    def hook_after_sd_init_before_load(self):
        super().hook_after_sd_init_before_load()
        self.maybe_stop()
        self.sd.add_status_update_hook(self.status_update_hook_func)

    def sample_step_hook(self, img_num, total_imgs):
        super().sample_step_hook(img_num, total_imgs)
        self.maybe_stop()
        self.update_status(
            "running", f"Generating images - {img_num + 1}/{total_imgs}")

    def sample(self, step=None, is_first=False):
        self.maybe_stop()
        total_imgs = len(self.sample_config.prompts)
        self.update_status("running", f"Generating images - 0/{total_imgs}")
        super().sample(step, is_first)
        self.maybe_stop()
        self.update_status("running", "Training")

    def save(self, step=None):
        self.maybe_stop()
        self.update_status("running", "Saving model")
        super().save(step)
        self.update_db_key("step", self.last_save_step)
        self.flush_ui_reporting()
        self.maybe_stop()
        self.update_status("running", "Training")
