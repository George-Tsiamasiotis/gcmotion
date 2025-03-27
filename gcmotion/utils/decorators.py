r"""
=========================
Useful utility decorators
=========================

In order to store the function's documentation so sphinx can see it, the
``__doc__`` attributes must be stored in the decorator itself. It is also
necessary to redefine the function signature when calling autodoc directives,
so the arguements and type hints are visible

"""


def _calls_counter(func):
    r"""Decorator counting how many times a function has been called."""

    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return func(*args, **kwargs)

    wrapped.__doc__ = func.__doc__

    wrapped.calls = 0
    return wrapped
