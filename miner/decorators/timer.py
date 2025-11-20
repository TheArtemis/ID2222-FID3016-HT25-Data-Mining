import time
import logging
from functools import wraps
from typing import Any
from collections.abc import Callable

# Add custom TIME logging level (between INFO=20 and DEBUG=10)
TIME_LEVEL = 15
logging.addLevelName(TIME_LEVEL, "TIME")


def time_log(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log a message with severity 'TIME'."""
    if self.isEnabledFor(TIME_LEVEL):
        self._log(TIME_LEVEL, message, args, **kwargs)


# Add the time method to Logger class
logging.Logger.time = time_log  # type: ignore[attr-defined]


def timer(
    logger: logging.Logger | str | None = None,
    level: int = TIME_LEVEL,
    active: bool = True,
) -> Callable:
    """
    A decorator that times the execution of a function.

    Args:
        logger: Optional logger instance or attribute name string. If a logger instance
                is provided, logs the timing information. If a string is provided (e.g., "logger"),
                it will look for that attribute on the first argument (self) at runtime.
                If None, automatically tries to find a "logger" attribute on the first argument
                (self) for methods, otherwise prints to stdout.
        level: Logging level to use when logger is provided. Defaults to TIME (15).
        active: If False, the decorator does nothing and returns the function unchanged.
                Defaults to True.

    Returns:
        Decorated function that times its execution, or the original function if active=False.

    Example:
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>>
        >>> @timer(logger=logger)
        >>> def my_function():
        >>>     # your code here
        >>>     pass
        >>>
        >>> # For methods, automatically uses self.logger if it exists
        >>> class MyClass:
        >>>     def __init__(self):
        >>>         self.logger = logging.getLogger(__name__)
        >>>
        >>>     @timer()  # Automatically uses self.logger
        >>>     def my_method(self):
        >>>         pass
        >>>
        >>> # Or without a logger (prints to stdout)
        >>> @timer()
        >>> def another_function():
        >>>     pass
        >>>
        >>> # Disable timing
        >>> @timer(active=False)
        >>> def production_function():
        >>>     pass
    """

    def decorator(func: Callable) -> Callable:
        if not active:
            return func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = end - start

            message = f"{func.__name__} took {elapsed:.4f} seconds"

            # Resolve logger at runtime
            actual_logger: logging.Logger | None = None
            if logger is not None:
                if isinstance(logger, str):
                    # If logger is a string, try to get it from the first argument (self)
                    if args and hasattr(args[0], logger):
                        actual_logger = getattr(args[0], logger)
                else:
                    actual_logger = logger
            else:
                # Default behavior: try to find "logger" attribute on self
                if args and hasattr(args[0], "logger"):
                    actual_logger = getattr(args[0], "logger")

            if actual_logger is not None:
                if level == TIME_LEVEL and hasattr(actual_logger, "time"):
                    actual_logger.time(message)  # type: ignore[attr-defined]
                else:
                    actual_logger.log(level, message)
            else:
                print(message)

            return result

        return wrapper

    return decorator
