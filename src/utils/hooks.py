from functools import wraps

# TODO(fdschmidt93): upstream to trident-core?
def processing_hooks(func):
    """
    Hooks for preprocessing that are pre- and appended to any built-in preprocessing function.

    Args:
        on_before_processing (:obj:`Callable`):
            Hook executed before the function call with the following function signature.

            .. code-block:: python

                def on_before_processing(inputs: dict, *args, **kwargs) -> dict:
                    ...
                    return inputs

        on_after_processing (:obj:`Callable`):
            Hook executed after the function call with the following function signature.

            .. code-block:: python

                def on_after_processing(inputs: dict, *args, **kwargs) -> BatchEncoding:
                    ...
                    return inputs
    """

    @wraps(func)
    def hooks(inputs: dict, *args, **kwargs):
        on_before_processing = kwargs.pop("on_before_processing", None)
        on_after_processing = kwargs.pop("on_after_processing", None)
        # avoid type checking for speed
        if on_before_processing is not None:
            inputs = on_before_processing(inputs, *args, **kwargs)
        inputs = func(inputs, *args, **kwargs)
        if on_after_processing is not None:
            inputs = on_after_processing(inputs, *args, **kwargs)
        return inputs

    return hooks
