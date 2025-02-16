r"""
Script/function that calculates (a single) fixed point of the GC Hamiltonian, based on a
single initial condition.
"""

from scipy.optimize import differential_evolution, fsolve
from collections import deque
from gcmotion.entities.profile import Profile
from gcmotion.utils.fp_system import system


# Function to locate a single fixed point
def fixed_point(
    method: str,
    initial_condition: list | tuple | deque,
    bounds: list,
    profile: Profile,
    known_thetas: bool = False,
    known_theta_value: float = 0,
) -> tuple[float, float]:
    r"""
    Function that finds a single fixed point of the GC Hamiltonian for a given profile
    (tokamak, particle information). It uses numerical solvers/methods "fsolve" or
    "differential evolution" to locate for which values of the :math:`\theta` and :math:`\psi`
    variables, their time derivatives become zero.


        Parameters
        ----------
        method : str
            Indicates which numerical method (solver) is used. Can be "fsolve" or "differential evolution".
        initial_condition : list | tuple | deque
            Initial condition passed into the numerical solver, in order to begin its search/
            iterative process.
        bounds : list
            Necessary to define a search region for the "differential evolution" numerical
            solver. Not used in "fsolve".
        profile : Profile
            Profile object that contains Tokamak and Particle information.
        known_thetas : bool, optional
            Boolean that indicates weather or not the LAR thetas are used (angles for which we
            know fixed oints occur). Defaults to False
        known_theta_value : float, optional
            Value of the LAR theta used. For this value the numerical solver will search for
            fixed :math:`\psi`. Defaults to 0.

        Returns
        -------
        tuple
            Fixed point (tuple) of the form (:math:`\theta`,:math:`\psi`) that essentially
            represents a solution of the numercial solver. :math:`\psi` is calculated in
            "NUMagnetic_flux" units.

    """

    # System of equations to be solved
    def fixed_point_system(vars):

        if known_thetas:
            psi = vars[0]
            theta = known_theta_value
        else:
            theta, psi = vars

        # We calculate the quantity theta_dot**2+psi_dot**2 or [theta_dot, psi_dot]
        # depending on the method
        system_result = system(theta=theta, psi=psi, profile=profile, method=method)

        if known_thetas and method == "fsolve":
            return system_result[0] ** 2 + system_result[1] ** 2

        return system_result

    if method == "differential evolution":

        result = differential_evolution(
            fixed_point_system,
            bounds,
            x0=initial_condition,
            tol=1e-9,
            atol=1e-15,
            maxiter=15_000,
            popsize=20,  # 15,
            mutation=(0.5, 1),  # (0.7, 1.5),
            recombination=0.7,  # 0.8,
            strategy="best1bin",  # "best2bin",
        )

        psi_solution = result.x[0] if known_thetas else result.x[1]
        theta_solution = known_theta_value if known_thetas else result.x[0]

    elif method == "fsolve":

        result = fsolve(
            fixed_point_system, x0=initial_condition, xtol=1e-10, maxfev=1_000, factor=0.1
        )

        psi_solution = result[0] if known_thetas else result[1]
        theta_solution = known_theta_value if known_thetas else result[0]

    else:
        print('method can be either "fsolve" or "differential evolution"')
        return

    return theta_solution, psi_solution
