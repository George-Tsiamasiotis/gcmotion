import sys
from pint import UnitRegistry

byte = UnitRegistry().byte


def _get_size(obj, seen=None):
    """Recursively finds size of objects.

    All of the code is gracefully borrowed from Wissam Jarjoui
    from `<https://goshippo.com/blog/measure-real-size-any-python-object>`_.

    Parameters
    ----------
    object : object
        The object to find the size of.
    seen : set
        Set that keeps track of object ids that have already been measured.
        Should be left to default since the fuction uses it when calling itself
        recursively to handle self-referential objects. Defaults to None.

    Returns
    -------
    int
        The size of the object in bytes.
    """

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([_get_size(v, seen) for v in obj.values()])
        size += sum([_get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += _get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(
        obj, (str, bytes, bytearray)
    ):
        try:
            size += sum([_get_size(i, seen) for i in obj])
        except TypeError:
            pass
    return size


def get_size(obj):
    r"""Quantifies _get_size() result and prints it."""
    print(f"{_get_size(obj) * byte:.4g~#P}")
