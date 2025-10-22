import casadi as cas


def get_end_effector_position_for_all_random(model, x_single) -> cas.MX:
    """
    Get the end-effector position for all random trials
    """
    nb_random = model.n_random
    ee_pos = cas.MX.zeros(2, nb_random)
    for i_random in range(nb_random):
        q_this_time = x_single[i_random * model.nb_q : (i_random + 1) * model.nb_q]
        ee_pos[:, i_random] = model.end_effector_position(q_this_time)
    return ee_pos


def get_end_effector_velocity_for_all_random(model, x_single) -> cas.MX:
    """
    Get the end-effector velocity for all random trials
    """
    nb_random = model.n_random
    ee_vel = cas.MX.zeros(2, nb_random)
    for i_random in range(nb_random):
        q_this_time = x_single[i_random * model.nb_q : (i_random + 1) * model.nb_q]
        qdot_this_time = x_single[model.q_offset + i_random * model.nb_q : model.q_offset + (i_random + 1) * model.nb_q]
        ee_vel[:, i_random] = model.end_effector_velocity(q_this_time, qdot_this_time)
    return ee_vel


def get_end_effector_for_all_random(model, x_single) -> tuple[cas.MX, cas.MX]:
    """
    Get the end-effector position and velocity for all random trials
    """
    ee_pos = get_end_effector_position_for_all_random(model, x_single)
    ee_vel = get_end_effector_velocity_for_all_random(model, x_single)
    return ee_pos, ee_vel
