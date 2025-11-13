import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import casadi as cas

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


def test_sample_task_from_circle():
    from ReachingLearning.LearningInternalDynamics.bayesian_inspired.utils import sample_task_from_circle

    home_position = np.array([-0.0212132, 0.445477])

    targets_starts = np.zeros((100, 2))
    targets_ends = np.zeros((100, 2))
    for i_sample in range(100):
        start, end = sample_task_from_circle()
        targets_starts[i_sample, :] = np.array(start).reshape(2, )
        targets_ends[i_sample, :] = np.array(end).reshape(2, )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(targets_starts[:, 0], targets_starts[:, 1], '.g')
    ax.plot(targets_ends[:, 0], targets_ends[:, 1], '.r')
    circ = plt.Circle(home_position, 0.15, fill=False, linestyle="-")
    ax.add_patch(circ)
    ax.axis("equal")
    plt.savefig("test_sample_task_from_circle.png")
    plt.show()

    dist_start = np.linalg.norm(targets_starts - home_position, axis=1)
    dist_end = np.linalg.norm(targets_ends - home_position, axis=1)
    npt.assert_array_less(dist_start, 0.15 + 1e-8)
    npt.assert_array_less(dist_end, 0.15 + 1e-8)


def test_dynamics():
    from ReachingLearning.LearningInternalDynamics.bayesian_inspired.spline_dynamics_parameters import SplineParametersDynamicsLearner, get_tau_opt
    from ReachingLearning.LearningInternalDynamics.bayesian_inspired.utils import get_the_real_dynamics

    real_forward_dyn, inv_mass_matrix_func, nl_effect_vector_func = get_the_real_dynamics()

    # 7612 is wisely chosen so that it not on the bound on any of the components
    x_7612 = np.array([1.22173048,   1.83259571, -24.43460953, -17.45329252])
    mus = np.array([0.5, 0.3, 0.1, 0.9, 0.3, 0.7])
    motor_noise = np.array([0.0, 0.0])

    tau = get_tau_opt(x_7612[:2].reshape(-1, 1), x_7612[2:].reshape(-1, 1), mus.reshape(-1, 1)).reshape(2, )

    real_value = np.array(real_forward_dyn(x_7612, tau, motor_noise))

    learner = SplineParametersDynamicsLearner(2, 0.0, enable_plotting=False)
    learner.spline_model["M11"] = lambda y: np.array(inv_mass_matrix_func(y)[0, 0])
    learner.spline_model["M12"] = lambda y: np.array(inv_mass_matrix_func(y)[0, 1])
    learner.spline_model["M22"] = lambda y: np.array(inv_mass_matrix_func(y)[1, 1])
    learner.spline_model["N1"] = lambda y: np.array(nl_effect_vector_func(y)[0])
    learner.spline_model["N2"] = lambda y: np.array(nl_effect_vector_func(y)[1])  # 0.0244125
    learner_value = learner.forward_dyn(x_7612, tau, motor_noise)

    np.repeat(np.array([[0.1, 0.2, 1.6, -4.1]]), 10000, 0)
    learner.spline_model["M11"] = lambda y: np.repeat(np.array(inv_mass_matrix_func(cas.reshape(y[7612, :], 4, 1))[0, 0]), 10000, 0).reshape(10000, )
    learner.spline_model["M12"] = lambda y: np.repeat(np.array(inv_mass_matrix_func(cas.reshape(y[7612, :], 4, 1))[0, 1]), 10000, 0).reshape(10000, )
    learner.spline_model["M22"] = lambda y: np.repeat(np.array(inv_mass_matrix_func(cas.reshape(y[7612, :], 4, 1))[1, 1]), 10000, 0).reshape(10000, )
    learner.spline_model["N1"] = lambda y: np.repeat(np.array(nl_effect_vector_func(cas.reshape(y[7612, :], 4, 1))[0]), 10000, 0).reshape(10000, )
    learner.spline_model["N2"] = lambda y: np.repeat(np.array(nl_effect_vector_func(cas.reshape(y[7612, :], 4, 1))[1]), 10000, 0).reshape(10000, )
    casadi_func = learner.casadi_forward_dyn_func()
    casadi_value = np.array(casadi_func(cas.DM(x_7612), cas.DM(mus)))

    npt.assert_almost_equal(real_value.reshape(-1, ), learner_value.reshape(-1, ), decimal=6)
    npt.assert_almost_equal(real_value.reshape(-1, ), casadi_value.reshape(-1, ), decimal=5)