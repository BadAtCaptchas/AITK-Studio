import unittest
import io
from unittest import mock

from toolkit.progress_bar import ToolkitProgressBar


class ToolkitProgressBarTest(unittest.TestCase):
    def test_pause_unpause_does_not_shift_elapsed_time_by_default(self):
        progress_bar = ToolkitProgressBar(total=1, file=io.StringIO())
        start_t = progress_bar.start_t
        last_print_t = progress_bar.last_print_t

        with mock.patch.object(progress_bar, "_now", side_effect=[10.0, 20.0]):
            progress_bar.pause()
            progress_bar.unpause()

        self.assertEqual(progress_bar.start_t, start_t)
        self.assertEqual(progress_bar.last_print_t, last_print_t)
        progress_bar.close()

    def test_pause_unpause_can_explicitly_exclude_elapsed_time(self):
        progress_bar = ToolkitProgressBar(total=1, file=io.StringIO())
        start_t = progress_bar.start_t

        with mock.patch.object(progress_bar, "_now", side_effect=[10.0, 20.0]):
            progress_bar.pause(exclude_elapsed=True)
            progress_bar.unpause()

        self.assertEqual(progress_bar.start_t, start_t + 10.0)
        self.assertEqual(progress_bar.last_print_t, 20.0)
        progress_bar.close()

    def test_update_is_suppressed_while_paused(self):
        progress_bar = ToolkitProgressBar(total=2, file=io.StringIO())
        progress_bar.pause()
        progress_bar.update(1)

        self.assertEqual(progress_bar.n, 0)
        progress_bar.unpause()
        progress_bar.update(1)
        self.assertEqual(progress_bar.n, 1)
        progress_bar.close()


if __name__ == "__main__":
    unittest.main()
