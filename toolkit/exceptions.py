class JobStopRequested(Exception):
    """Expected stop requested by the UI or queue manager."""

    def __init__(self, message: str = "Job stopped", return_to_queue: bool = False):
        super().__init__(message)
        self.return_to_queue = return_to_queue
