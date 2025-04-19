import importlib
import traceback


def str_to_callable(dotted_path: str):
    """
    Converts a dotted path string to a callable Python object.

    Args:
        dotted_path (str): Dotted path to the callable, e.g. 'matrix.job.job_utils.echo'

    Returns:
        callable: The resolved callable object

    Raises:
        ValueError: If the path is malformed, module can't be imported, or the object is not callable.
    """
    try:
        module_path, func_name = dotted_path.rsplit(".", 1)
    except ValueError:
        raise ValueError(
            f"Invalid path '{dotted_path}': Must be in 'module.attr' format."
        )

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Module '{module_path}' could not be imported. Original error:\n{e}"
        )

    try:
        obj = getattr(module, func_name)
    except AttributeError:
        raise ValueError(f"'{func_name}' not found in module '{module_path}'.")

    if not callable(obj):
        raise ValueError(f"'{dotted_path}' resolved to a non-callable object.")

    return obj
