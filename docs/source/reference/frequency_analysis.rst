.. currentmodule:: gcmotion

==========================
gcmotion.FrequencyAnalysis
==========================

The FrequencyAnalysis class iterates through (μ, Pζ, Ε) values upon a given
Tokamak, and finds the ωθ, ωζ frequencies and their ratio qkinetic by searching
for contours.

.. autoclass:: FrequencyAnalysis
   :class-doc-from: class
   :members: start, to_dataframe, scatter
   :member-order: bysource


Notes on Input shapes
---------------------

The algorithm supports 3 modes, which are activated depending on the shape of
the input arrays:

1. Cartesian Mode: Activated by passing 3 1D arrays.
    The algorithm takes all combinations (cartesian product) of every array
    entry. If you want to iterate through only 1 COM value, use
    np.array([<value>]).

2. Matrix Mode: Activated by passing 3 2D arrays, **with the same shape**.
    The algorithm creates triplets of COMs by stacking the 3 arrays. Each
    triplet is defined as (muspan[i,j], Pzetaspan[i,j], Espan[i,j]), where
    0<=i<nrows and 0<=j<ncols. Useful when the grid is not orthogonal, for
    example when analysing a certain :math:`P_\zeta - E` domain of the
    parabolas.

    Note:
    If we know that our grid is orthogonal, it's much faster to use a row from each span array and use cartesian mode, since its significantly faster, especially when iterating through a lot of energies.

3. Dynamic minimum energy Mode: Activated by only passing muspan and Pzetspan as 1D arrays. 
    The algorithm finds the minimum vaule of the energy grid for every
    (:math:`\mu`, :math:`P_\zeta`) pair, which is always found at an O-point,
    and slowly increments it until it finds 1 trapped orbit. This orbit's
    frequency is (very close to) the O-point frequency, which is always the
    highest frequency of this specific family of trapped orbits. This frequency
    defines the maximum frequency with which the particles resonate, with 0
    being the minimum (separatrix). This mode can only find the O-point
    frequency on the total minumum energy. If more O-points are present, then
    we must use method 4.

.. todo::

    4. Dynamic O-point minimum energy Mode:

Algorithm
---------

Each contour represents a specific family of orbits represented by the same 3
Constants of Motion, and differing only in their initial conditions (however, a
single triplet may correspond to more that 1 contour). By exploiting the fact
that our poloidal angle is in fact the boozer theta, the area contained within
the contour is equal to :math:`2πJ_\theta`, where :math:`J_\theta` the corresponding 
action variable. We then use the definitions:

.. math::

    \omega_\theta = \dfrac{dE}{dJ_\theta} \\
    q_{kin} = - \dfrac{dJ_\theta}{dJ_\zeta} = -\dfrac{dJ_\theta}{dP_\zeta} \\
    \omega_\zeta = q_{kin} * \omega_\theta \\

The algorithm follows these steps, regardless of the "scanning" method
(cartesian, matrix, dymanic energy minimum, ...), since the only thing they
change is the way the triplets (μ, Ε, Ρζ) are created:

The area is found using the `shoelace formula <https://en.wikipedia.org/wiki/Shoelace_formula>`_

1. For a triplet, iterate first through all the energies, then Ρζ and then μ:

    >>> for mu in muspan:
    >>>     ...
    >>>     for pzeta in pzetaspan:
    >>>         ...
    >>>         create_main_contour()
    >>>         ...
    >>>         for energy in energyspan:
    >>>             ...

    Since all orbits with the same μ and Ρζ share the same slice, we can
    calculate the Main Contour once and use it for all energies. As for the μ
    and Ρζ loops, they only update the Profile's μ and Ρζ and are completely symmetrical.

    The Main Contour is always plotted with θ from -2π to 2π, while ψ is
    defined by the user. This ensures that all orbits are present, even trapped
    orbits around θ=π. These orbits also appear twice, which leads to
    calculating the same thing two times, but the performance impact is not
    worth the added complexity needed to avoid it...

    For matrix mode, we must create the main contour for *every* energy as
    well, since the grid is generally not orthogonal.

    Note that we do not use matplotlib's contour methods, but contourpy's
    ContourGenerator. Not only this is much faster and more memory efficient,
    but it also gives more control when extracting contour lines.

2. For every energy now, we extract all contour lines the generator found for
that energy. The number of contour lines can be 0 up to 4-5, depending on the
equilibrium and energy level.

    The contour lines are returned as a list of (N, 2) numpy arrays.

3. For every line we create a ContourOrbit object. This object represents a
single family orbit. It contains methods for validating itself and classifying
the orbit type, and also stores all calculated quantities and frequencies.

4. We calculate the bounding box of the orbit (e.g. the smallest rectangle
fully containing the orbit who's sides are parallel to the x and y axes).

5. We classify the orbit as trapped or passing. This is immediately calculated
by checking whether the bounding box touches both left and right walls
(passing) or not (trapped).

If the orbit is passing, we further classify it as co- or cu-passing. For this 
we use the approximation that if rho>0 on all vertices then the orbit is 
co-passing, else cu-passing.

5. We validate the orbit, by checking 2 things:

    - The orbit is fully contained within the contour limits and doesn't get
      cutoff. This is True if its bounding box does not touch any of the upper 
      or lower contour walls.

    - The orbit is not cutoff-trapped, e.g. a trapped orbit that gets cut off
      by the left and right contour walls. The orbit is cutoff-trapped if its
      bounding box touches the left or the right wall, but not both.

    These 2 simple conditions are enough to fully validate the orbit. If one of
    those checks fails, we discard the orbit.

6. Calculate the orbit's frequencies. The idea is to generate 2 adjacent local
contours by slightly increasing and decreasing one of the constants Pζ or E and
calculating the same orbit in these slightly different contours. We can then
calculate their Jθs. Their difference would be dJθ, while dPζ or dE is defined
by the user. Using the above definitions, we can calculate each frequency using
these values. This is basically calculating the derivative locally.

    - Since the Main Contour is calculated in the θ-ψ plane, we need to convert
      the ContourOrbit's vertices' y coordinate to Pθ to correctly calculate
      Jθ. No need to convert the whole ψ-grid to Ρθ-grid each time.

    - If the main orbit has enough vertices as to calculate its Jθ accurately
      enough, we can avoid creating a 2nd contour by using the main orbit
      itself as one of the adjacent contours. This is what happens with most
      orbits. However, for low-energy trapped orbits, the orbit dimensions are
      comparable to the contour's grid spacing, making the contour line jagged.
      By creating 2 adjacent local contours we circumvent this problem, since
      the orbit in the adjacent contour has much more vertices, and the main
      orbit's Jθ is not calculated at all.

    - If at any point of the calculation an invalid contour line is found, the
      calculation is aborted. The number of orbits lost by this restriction
      however is negligible.

Advantages
----------

1. The algorithm is very close to the theoretical formulation of the Action-Angle 
theory, and uses a purely geometrical way of calculating frequencies, without the 
need for particle tracking.

2. There is no restriction as to how close or how far the array spans need to be 
spaced, since each triplet's frequencies are calculated independently. This makes 
it possible to hold one or two of the COMs constant and iterate through a range of 
the the third COM. Moreover, we can use a more dense array for areas we expect 
more trapped orbits and a less dense one for areas with more passing orbits,
since the Hamiltonian tends to be "flatter" in areas around O-points.

Caveats
-------

1. The approximation of rho in the co-/cu-passing classification is usually
correct, except for some orbits exceptionally close to the separatrix. At the
presence of more O-points, new families of passing orbits may be created, which
cannot be classified this way. Those orbits have the flag "undefined".

2. When calculating the Jθ for a passing orbit, we must add the points [-2π, 0]
and [0, 2π] to its vertices. Since the order (left-to-right or right-to-left)
of the vertices returned by the ContourGenerator is effectively random (but is
always one of the two), we must first check their direction and then add the 
points in the right order. We must also divide by 2, since the contour is 
calculated in [-2π, 2π].

3. When calculating same-level orbits in the adjacent contours, the
ContourGenerator may return more than 1 orbits, in a random order. This happens
more often than not, since for most passing orbits, there is a corresponding
copassing orbit with the same COMs. It can also happen with trapped orbits when
more than 1 O-points are present. This problem can be solved by comparing the
distances of the orbits' bounding boxes (which are already calculated at this
point) from the main orbit's bounding box, and picking the closest one.
Specifically, we compare the distances of the bottom left corner.

4. Does not work under the presence of perturbations.
