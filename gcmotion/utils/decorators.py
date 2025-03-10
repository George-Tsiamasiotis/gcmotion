r"""Useful utility decorators"""


def _calls_counter(func):
    r"""Decorator counting how many times a function has been called."""

    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return func(*args, **kwargs)

    wrapped.calls = 0
    return wrapped
