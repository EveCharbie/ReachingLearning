import casadi as cas


def get_states_from_muscle_lengths(arm_model: "ArmModel", muscle_lengths):
    """
    Given mono-articular muscle lengths, it compute the corresponding joint angles (q) by averaging the estimates from
    the agonist/antagonist pair.
        - Shoulder angle from posterior deltoid and anterior deltoid
        - Elbow angle from brachialis and lateral triceps
    """
    # Shoulder

    arm_model.dM_coefficients = np.array(
        [
            [0, 0, 0.0100, 0.0300, -0.0110, 1.9000],
            [0, 0, 0.0100, -0.0190, 0, 0.0100],
            [0.0400, -0.0080, 1.9000, 0, 0, 0.0100],
            [-0.0420, 0, 0.0100, 0, 0, 0.0100],
            [0.0300, -0.0110, 1.9000, 0.0320, -0.0100, 1.9000],
            [-0.0390, 0, 0.0100, -0.0220, 0, 0.0100],
        ]
    )

    arm_model.a_shoulder = arm_model.dM_coefficients[:, 0]
    arm_model.b_shoulder = arm_model.dM_coefficients[:, 1]
    arm_model.c_shoulder = arm_model.dM_coefficients[:, 2]
    arm_model.a_elbow = arm_model.dM_coefficients[:, 3]
    arm_model.b_elbow = arm_model.dM_coefficients[:, 4]
    arm_model.c_elbow = arm_model.dM_coefficients[:, 5]

    l_full = (
        arm_model.a_shoulder * theta_shoulder
        + arm_model.b_shoulder * cas.sin(arm_model.c_shoulder * theta_shoulder) / arm_model.c_shoulder
        + arm_model.a_elbow * theta_elbow
        + arm_model.b_elbow * cas.sin(arm_model.c_elbow * theta_elbow) / arm_model.c_elbow
    )

    elbow = (
        +arm_model.a_elbow * theta_elbow
        + arm_model.b_elbow * cas.sin(arm_model.c_elbow * theta_elbow) / arm_model.c_elbow
    )

    return cas.vertcat(q1, q2)
