import numpy as np
import numpy.testing as npt
import biorbd
import pytest

from ReachingLearning import get_states_from_muscle_lengths


@pytest.mark.parametrize("elbow_angle",
                         [
                             # 15,
                             # 30,
                             # 45,
                             # 60,
                             # 75,
                             # 90,
                             # 105,
                             # 120,
                             135,
                             150,
                             165,
                         ])
@pytest.mark.parametrize("shoulder_angle",
                         [
                             15,
                             30,
                             45,
                             60,
                             75,
                             90
                        ])
def test_get_states_from_muscle_lengths(elbow_angle, shoulder_angle):

    biorbd_model = biorbd.Biorbd("../ReachingLearning/StochasticOptimalControl/models/arm_model.bioMod")

    # Generate the data
    q = np.array([shoulder_angle * np.pi / 180, elbow_angle * np.pi / 180])
    print(q)
    muscle_lengths = biorbd_model.muscles.muscle_tendon_length(q)

    # Test the functions and its inverse
    q_computed = get_states_from_muscle_lengths(muscle_lengths)
    npt.assert_almost_equal(q, np.array(q_computed).reshape((2,)), decimal=5)

    # # Test that the noised version is not too bad
    # muscle_noise_magnitude = 0.002
    # noise = np.random.normal(0, muscle_noise_magnitude, len(muscle_lengths))
    # q_estimated = get_states_from_muscle_lengths(muscle_lengths + noise)
    # estimation_error = q - np.array(q_estimated).reshape((2,))
    # npt.assert_array_less(estimation_error, 10 * np.pi / 180)  # less than 10 degrees of error
    #
    # print(f"Shoulder: {shoulder_angle} deg, Elbow: {elbow_angle} deg -> OK")
