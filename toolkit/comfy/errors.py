class ComfyError(RuntimeError):
    pass


class ComfyConfigError(ComfyError):
    pass


class ComfyWorkflowError(ComfyError):
    pass
