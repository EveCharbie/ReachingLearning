import numpy as np
import numpy.testing as npt

from ReachingLearning import ArmModel, get_states_from_muscle_lengths


def test_get_states_from_muscle_lengths():

    elbow_angles = [15 * np.pi / 180, 30 * np.pi / 180, 45 * np.pi / 180, 60 * np.pi / 180]
    shoulder_angles = [15 * np.pi / 180, 30 * np.pi / 180, 45 * np.pi / 180, 60 * np.pi / 180]

    arm_model = ArmModel(sensory_noise_magnitude=0, motor_noise_magnitude=0, force_field_magnitude=0)

    for shoulder in shoulder_angles:
        for elbow in elbow_angles:

            # Generate the data
            q = np.array([shoulder, elbow])
            muscle_lengths = arm_model.get_muscle_length(q)

            # Test the functions and its inverse
            q_computed = get_states_from_muscle_lengths(arm_model, muscle_lengths)
            npt.assert_almost_equal(q, q_computed, decimal=5)

            # Test that the noised version ios not too bad
            noise = np.random.normal(0, 0.002, muscle_lengths.shape)
            q_estimated = get_states_from_muscle_lengths(arm_model, muscle_lengths + noise)
            npt.assert_almost_equal(q, q_estimated, decimal=2)
