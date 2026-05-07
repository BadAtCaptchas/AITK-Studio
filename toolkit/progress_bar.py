from tqdm import tqdm
import time


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused = False
        self.last_time = self._now()
        self.exclude_elapsed_when_paused = False

    def _now(self):
        return time.time()

    def pause(self, exclude_elapsed=False):
        if not self.paused:
            self.paused = True
            self.last_time = self._now()
            self.exclude_elapsed_when_paused = exclude_elapsed

    def unpause(self):
        if self.paused:
            self.paused = False
            if self.exclude_elapsed_when_paused and hasattr(self, 'start_t'):
                cur_t = self._now()
                self.start_t += cur_t - self.last_time
                self.last_print_t = cur_t
            self.exclude_elapsed_when_paused = False

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)
