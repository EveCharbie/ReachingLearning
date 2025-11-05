import numpy as np
import numpy.testing as npt

from ReachingLearning.LearningInternalDynamics.bayesian_inspired.utils import get_the_real_dynamics


def test_get_the_real_dynamics():
    real_dynamics, inv_mass_matrix_func, nl_effect_vector_func = get_the_real_dynamics()

    x_eval = np.array([0.1, 0.2, 1.6, -4.1])
    u_eval = np.array([0.5, -0.3])
    motor_noise = np.array([0, 0])

    dyn_eval = np.array(real_dynamics(x_eval, u_eval, motor_noise))
    inv_mass_eval = np.array(inv_mass_matrix_func(x_eval))
    nl_effect_eval = np.array(nl_effect_vector_func(x_eval))
    reconstructed_dyn = np.array(inv_mass_eval @ (u_eval.reshape(2, 1) - nl_effect_eval))

    # Please note that in biorbd the NLEffects contains the Coriolis + Gravity effects and the multiplication by qdot
    # Check qdot = dq
    npt.assert_almost_equal(dyn_eval[:2].reshape(2, ), x_eval[2:].reshape(2, ))
    # Check qddot = dqdot
    npt.assert_almost_equal(dyn_eval[2:].reshape(2, ), reconstructed_dyn.reshape(2, ))
    # Check that the mass matrix is invertible
    npt.assert_almost_equal(inv_mass_eval @ np.linalg.inv(inv_mass_eval), np.eye(2))
    # Check that the mass matrix is symmetric
    npt.assert_almost_equal(inv_mass_eval[0, 1], inv_mass_eval[1, 0])
