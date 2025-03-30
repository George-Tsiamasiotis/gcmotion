# =====================TESTING FUNCTIONS=============================


def test_bifurcation_instantiation(bifurcation_output_energies):
    r"""Tests bifurcation instantiation essentally."""
    bifurcation_output, N = bifurcation_output_energies

    # Unpack bifurcation output
    X_thetas = bifurcation_output["X_thetas"]
    X_psis = bifurcation_output["X_psis"]
    O_thetas = bifurcation_output["O_thetas"]
    O_psis = bifurcation_output["O_psis"]
    num_of_XP = bifurcation_output["num_of_XP"]
    num_of_OP = bifurcation_output["num_of_OP"]
    X_energies = bifurcation_output["X_energies"]
    O_energies = bifurcation_output["O_energies"]

    assert len(X_thetas) == len(O_psis) == N
    assert len(O_thetas) == len(X_psis) == N
    assert len(X_energies) == len(num_of_OP) == N
    assert len(O_energies) == len(num_of_XP) == N


def test_bifurcation_kwargs_passing(bifurcation_output_no_energies):
    r"""Tests weather some input arguments are passed correctly."""
    bifurcation_output = bifurcation_output_no_energies

    X_energies = bifurcation_output["X_energies"]
    O_energies = bifurcation_output["O_energies"]

    assert len(X_energies) == len(O_energies) == 0
