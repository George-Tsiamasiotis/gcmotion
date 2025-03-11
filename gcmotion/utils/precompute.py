import numpy as np
from scipy.special import hyp2f1

from gcmotion.configuration.scripts_configuration import PrecomputedConfig

config = PrecomputedConfig()


def precompute_hyp2f1(n, zspan):
    a = b = 1 / n
    c = 1 + 1 / n

    zs = np.linspace(zspan[0], zspan[1], config.hyp2f1_density)
    values = hyp2f1(a, b, c, zs)

    return zs, values
