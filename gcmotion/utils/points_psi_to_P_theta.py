r"""
Script/function that turns lists that contain elements of the form [:math:`\theta`, :math:`\psi`] 
to elements of the form[:math:`\theta`, :math:`P_{theta}`]. 
"""

from collections import deque
import numpy as np
from gcmotion.entities.profile import Profile


def points_psi_to_P_theta(points: np.ndarray | list | deque, profile: Profile):
    r"""
    Takes in points list/array/deque of the form [:math:`\theta`, :math:`\psi`] returns [:math:`\theta`, :math:`\P_{theta}`]
    after calculating the :math:`P_{theta}` corresponding to each :math:`\psi`

        Parameters
        ----------
        points : np.ndarray, list, deque
            Iterable (np.ndarray, list, deque) that contains point of the form [:math:`\theta`, :math:`\psi`].
        profile : Profile
            Profile object that contains Tokamak and Particle information.

        Returns
        -------

        new_points : np.ndarray, list, deque
            List that contains points of the form [:math:`\theta`, :math:`P_{theta}`].


    """

    # Define a Quantity object. Will be needed later
    Q = profile.Q

    new_points = points.copy()

    if isinstance(points, np.ndarray):
        new_points = deque(new_points)

    for i in range(len(new_points)):

        theta, psi = new_points[i]

        P_thetaNU = profile.findPtheta(
            psi=Q(abs(psi), "NUMagnetic_flux"), units="NUCanonical_momentum"
        )

        new_points[i] = [theta, P_thetaNU.m]

    return new_points
