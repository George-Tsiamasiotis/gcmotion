import gcmotion as gcm
from gcmotion.utils.get_size import _get_size


def test_get_size_functionality(simple_tokamak):
    _get_size(simple_tokamak)
    gcm.get_size(simple_tokamak)
