import casadi as cas

from .utils import ExampleType
from .penalty_utils import get_end_effector_for_all_random


def reach_target_consistently(model, x_single, example_type) -> cas.MX:
    """
    Encourage the end-effector position and velocity deviation (like std) to be small at the end of the movement
    across all random trials.
    a.k.a. "minimize the outcome/performance variations"
    """
    nb_random = model.n_random
    ee_pos, ee_vel = get_end_effector_for_all_random(model, x_single)
    ee_pos_mean = cas.sum2(ee_pos) / nb_random
    ee_vel_mean = cas.sum2(ee_vel) / nb_random

    if example_type == ExampleType.CIRCLE:
        deviations = cas.sum1((cas.sum2((ee_pos - ee_pos_mean) ** 2) + cas.sum2((ee_vel - ee_vel_mean) ** 2)))
    else:
        deviations = cas.sum2((ee_pos[1] - ee_pos_mean[1]) ** 2) + cas.sum2((ee_vel[1] - ee_vel_mean[1]) ** 2)
    return deviations


def minimize_stochastic_efforts(model, x_single, u_single, noise_single) -> cas.MX:
    muscle_activations = u_single[: model.nb_muscles]
    k_fb = u_single[model.nb_muscles : model.nb_muscles + model.nb_q * model.n_references]
    k_fb = model.reshape_vector_to_matrix(k_fb, model.matrix_shape_k_fb)
    ref_fb = u_single[model.nb_muscles + model.nb_q * model.n_references :]
    sensory_noise = noise_single[model.nb_q : model.nb_q + model.n_references]

    tau_computed = cas.MX.zeros(2, model.n_random)
    for i_random in range(model.n_random):
        q_this_time = x_single[i_random * model.nb_q : (i_random + 1) * model.nb_q]
        qdot_this_time = x_single[model.q_offset + i_random * model.nb_q : model.q_offset + (i_random + 1) * model.nb_q]

        tau_computed[:, i_random] = model.collect_tau(
            q_this_time, qdot_this_time, muscle_activations, k_fb, ref_fb, sensory_noise
        )

    return cas.sum1(cas.sum2(tau_computed**2))


def minimize_gains(model, u_single):
    k_fb = u_single[model.nb_muscles : model.nb_muscles + model.nb_q * model.n_references]
    return cas.sum1(k_fb**2)
