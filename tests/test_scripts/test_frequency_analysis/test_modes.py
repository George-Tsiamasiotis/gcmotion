import pytest
import numpy as np
import gcmotion as gcm


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.parametrize(
    "freq_tokamaks",
    [
        "analytical",
        # "numerical",
    ],
    indirect=True,
)
@pytest.mark.parametrize("calculate_qkinetic", [True, False])
@pytest.mark.parametrize("calculate_omega_theta", [True, False])
# @pytest.mark.parametrize("cocu_classification", [True, False])
# @pytest.mark.parametrize("skip_trapped", [True, False])
# @pytest.mark.parametrize("skip_passing", [True, False])
@pytest.mark.parametrize("max_vertices_method_switch", [10, np.inf])
class TestModes:

    @staticmethod
    def test_cartesian(
        freq_tokamaks,
        calculate_qkinetic,
        calculate_omega_theta,
        # cocu_classification,
        # skip_trapped,
        # skip_passing,
        max_vertices_method_switch,
    ):

        # This span can find all types of orbits
        muspan = np.array([1e-5])
        Pzetaspan = np.array([-0.04])
        Espan = np.array([7.8e-6, 4e-5])

        freq = gcm.FrequencyAnalysis(
            tokamak=freq_tokamaks,
            psilim=(1e-7, 1.2),
            muspan=muspan,
            Pzetaspan=Pzetaspan,
            Espan=Espan,
            main_contour_density=60,
            local_contour_density=60,
            calculate_qkinetic=calculate_qkinetic,
            calculate_omega_theta=calculate_omega_theta,
            # cocu_classification=cocu_classification,
            # skip_trapped=skip_trapped,
            # skip_passing=skip_passing,
            max_vertices_method_switch=max_vertices_method_switch,
        )

        freq.__str__()

        freq.start(pbar=False)

        freq.__str__()
        freq.to_dataframe(True)

        assert freq.mode == "cartesian"
        assert freq.triplets_num == 2

    @staticmethod
    def test_matrix(
        freq_tokamaks,
        calculate_qkinetic,
        calculate_omega_theta,
        # cocu_classification,
        # skip_trapped,
        # skip_passing,
        max_vertices_method_switch,
    ):

        # This span can find all types of orbits
        muspan = np.array(
            (
                [1e-5, 2e-5],
                [1e-5, 3e-5],
            )
        )
        Pzetaspan = np.array(
            (
                [-0.04, -0.03],
                [-0.02, -0.01],
            )
        )
        Espan = np.array(
            (
                [7.6e-6, 3e-5],
                [7.8e-6, 4e-5],
            )
        )

        freq = gcm.FrequencyAnalysis(
            tokamak=freq_tokamaks,
            psilim=(1e-7, 1.2),
            muspan=muspan,
            Pzetaspan=Pzetaspan,
            Espan=Espan,
            main_contour_density=60,
            local_contour_density=60,
            calculate_qkinetic=calculate_qkinetic,
            calculate_omega_theta=calculate_omega_theta,
            # cocu_classification=cocu_classification,
            # skip_trapped=skip_trapped,
            # skip_passing=skip_passing,
            max_vertices_method_switch=max_vertices_method_switch,
        )

        freq.__str__()

        freq.start(pbar=False)

        freq.__str__()
        freq.to_dataframe(True)

        assert freq.mode == "matrix"
        assert freq.triplets_num == 4

    @staticmethod
    def test_dynamic_energy_minimum(
        freq_tokamaks,
        calculate_qkinetic,
        calculate_omega_theta,
        # cocu_classification,
        # skip_trapped,
        # skip_passing,
        max_vertices_method_switch,
    ):

        muspan = np.array([1e-5])
        Pzetaspan = np.array([-0.04])

        freq = gcm.FrequencyAnalysis(
            tokamak=freq_tokamaks,
            psilim=(1e-7, 1.2),
            muspan=muspan,
            Pzetaspan=Pzetaspan,
            Espan=None,
            main_contour_density=60,
            local_contour_density=60,
            calculate_qkinetic=calculate_qkinetic,
            calculate_omega_theta=calculate_omega_theta,
            # cocu_classification=cocu_classification,
            # skip_trapped=False,
            # skip_passing=skip_passing,
            max_vertices_method_switch=max_vertices_method_switch,
        )

        freq.__str__()

        freq.start(pbar=False)

        freq.__str__()
        freq.to_dataframe(True)

        assert freq.mode == "dynamicEmin"
        assert freq.trapped_orbits_num >= 1
