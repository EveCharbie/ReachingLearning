"""
This file was used to create the arm model used in the reaching learning example.
1. The inertia characteristics are extracted from Li & Todorov 2004.
2. The muscle origins, insertions, and via points are eye balled from the figure in Li & Todorov 2004.
3. The muscle via points positions are replaced by fitting the muscle lever arms (Livet et al. 2022) from Li & Todorov 2004.
4. The muscle origin and insertion positions are adjusted along the muscle trajectories (Livet et al. 2022) to match the muscle lengths from Li & Todorov 2004.
"""

import logging
import numpy as np
import casadi as cas

from biobuddy import (
    MuscleType,
    MuscleStateType,
    BiomechanicalModelReal,
    Rotations,
    RotoTransMatrix,
    SegmentReal,
    MeshFileReal,
    SegmentCoordinateSystemReal,
    RangeOfMotion,
    Ranges,
    InertiaParametersReal,
    MarkerReal,
    MuscleGroupReal,
    MuscleReal,
    ViaPointReal,
)

_logger = logging.getLogger(__name__)


def lever_arm_estimation(a, b, c, theta):
    return a + b * np.cos(c * theta)

def compute_lever_arm_squared(line_p1, line_p2, point):
    """
    Compute the lever arm (perpendicular distance) from a point to a line defined by two points in 2D.
    """
    # Vector from line_p1 to line_p2
    line_vec = line_p2 - line_p1
    # Vector from line_p1 to the point
    point_vec = point - line_p1
    # Project point_vec onto line_vec
    distance_to_line_perp_point = line_p1 + cas.dot(point_vec, line_vec) / cas.dot(line_vec, line_vec) * line_vec
    # Distance
    lever_arm_squared = cas.sum1((point - distance_to_line_perp_point) ** 2)
    return lever_arm_squared

def muscle_length_estimation(a_shoul, b_shoul, c_shoul, a_elbow, b_elbow, c_elbow, theta_shoul, theta_elbow):
    return (a_shoul * theta_shoul + b_shoul * np.sin(c_shoul * theta_shoul) / c_shoul +
            a_elbow * theta_elbow + b_elbow * np.sin(c_elbow * theta_elbow) / c_elbow)


def find_via_point_that_matches_lever_arm(model: BiomechanicalModelReal):
    """
    Step 3
    """
    nb_states = 50
    q = np.zeros((model.nb_q, nb_states))
    for i_dof in range(model.nb_q):
        # q[i_dof, :] = np.linspace(model.get_dof_ranges[i_dof].min_bound[0], model.q_ranges[i_dof].max_bound[0], nb_states)
        q[i_dof, :] = np.linspace(0, np.pi, nb_states)

    # Coefficients from Van Wouwe et al. 2022
    #       a_shoul,  b_shoul,   c_shoul, a_elbow, b_elbow, c_elbow
    dM_coefficients = np.array(
        [
            [0,       0,         0.001,   0.0300, -0.0110, 1.9000],   # brachialis = elbow flexor
            [0,       0,         0.001,  -0.0190,  0,      0.001],        # lateral triceps = elbow extensor
            [0.0400, -0.0080,    1.9000,  0,       0,      0.001],        # anterior deltoid = shoulder flexor
            [-0.0420, 0,         0.001,   0,       0,      0.001],        # posterior deltoid = shoulder extensor
            [0.0300, -0.0110,    1.9000,  0.0320, -0.0100, 1.9000],   # biceps = shoulder and elbow flexor
            [-0.0390, 0,         0.001,  -0.0220,  0,      0.001],        # long triceps = shoulder and elbow extensor
        ]
    )
    muscle_ranges = np.array([
        [1.1, 0.6],  # brachialis
        [0.8, 1.25], # lateral triceps
        [1.2, 0.7],  # anterior deltoid
        [0.7, 1.1],  # posterior deltoid
        [1.1, 0.6],  # biceps
        [0.85, 1.2], # long triceps
    ]).T


    lm_0 = muscle_length_estimation(
        dM_coefficients[:, 0],
        dM_coefficients[:, 1],
        dM_coefficients[:, 2],
        dM_coefficients[:, 3],
        dM_coefficients[:, 4],
        dM_coefficients[:, 5],
        0,
        0,
    )
    lm_180 = muscle_length_estimation(
        dM_coefficients[:, 0],
        dM_coefficients[:, 1],
        dM_coefficients[:, 2],
        dM_coefficients[:, 3],
        dM_coefficients[:, 4],
        dM_coefficients[:, 5],
        np.pi,
        np.pi,
    )
    muscle_length_multiplier = (muscle_ranges[0, :] - muscle_ranges[1, :]) / (lm_0 - lm_180)
    muscle_length_base = muscle_ranges[0, :]

    shoulder_lever_arms = np.zeros((6, nb_states))
    elbow_lever_arms = np.zeros((6, nb_states))
    for i_state in range(nb_states):
        shoulder_lever_arms[:, i_state] = lever_arm_estimation(dM_coefficients[:, 0], dM_coefficients[:, 1], dM_coefficients[:, 2], q[0, i_state])  # shoulder
        elbow_lever_arms[:, i_state] = lever_arm_estimation(dM_coefficients[:, 3], dM_coefficients[:, 4], dM_coefficients[:, 5], q[1, i_state])  # elbow
    lever_arm_to_match = np.vstack((
        elbow_lever_arms[0, :],  # brachialis
        elbow_lever_arms[1, :],  # lateral triceps
        shoulder_lever_arms[2, :], # anterior deltoid
        shoulder_lever_arms[3, :], # posterior deltoid
        shoulder_lever_arms[4, :], elbow_lever_arms[4, :], # biceps
        shoulder_lever_arms[5, :], elbow_lever_arms[5, :], # long triceps
    ))
    muscle_length_to_match = np.zeros((6, nb_states))
    for i_state in range(nb_states):
        muscle_length_to_match[:, i_state] = muscle_length_base + muscle_length_multiplier * muscle_length_estimation(
            dM_coefficients[:, 0],
            dM_coefficients[:, 1],
            dM_coefficients[:, 2],
            dM_coefficients[:, 3],
            dM_coefficients[:, 4],
            dM_coefficients[:, 5],
            q[0, i_state],
            q[1, i_state],
        )

    # Get the RT in global for all n_states at once
    shoulder_position = np.zeros((2, nb_states))
    global_rt = model.forward_kinematics(q)
    elbow_position = np.zeros((2, nb_states))
    for i_state in range(nb_states):
        elbow_position[:, i_state] = global_rt["lower_arm"][i_state].translation[:2]

    # Initialize problem
    nb_via_points = model.nb_via_points
    nb_muscles = model.nb_muscles
    via_point_x = cas.MX.sym("via_point_x", nb_via_points, 1)
    via_point_y = cas.MX.sym("via_point_y", nb_via_points, 1)
    origin_x = cas.MX.sym("origin_x", 6, 1)
    origin_y = cas.MX.sym("origin_y", 6, 1)
    insertion_x = cas.MX.sym("insertion_x", 6, 1)
    insertion_y = cas.MX.sym("insertion_y", 6, 1)
    objective = 0

    # Test joint configurations
    for i_state in range(nb_states):

        # Match the lever_arm
        brachialis_lever_arm = compute_lever_arm_squared(
            line_p1 = cas.vertcat(via_point_x[0], via_point_y[0]),  # via_point_position
            line_p2 = cas.vertcat(insertion_x[0], insertion_y[0]),  # insertion_position
            point = elbow_position[:, i_state],  # elbow_position
        )
        triceps_lateral_lever_arm = compute_lever_arm_squared(
            line_p1 = cas.vertcat(via_point_x[1], via_point_y[1]),   # via_point_position
            line_p2 = cas.vertcat(insertion_x[1], insertion_y[1]),  # insertion_position
            point = elbow_position[:, i_state],  # elbow_position
        )
        deltoid_anterior_lever_arm = compute_lever_arm_squared(
            line_p1 = cas.vertcat(origin_x[2], origin_y[2]),   # origin_position
            line_p2=cas.vertcat(insertion_x[2], insertion_y[2]),  # insertion_position
            point = shoulder_position[:, i_state],  # shoulder_position
        )
        deltoid_posterior_lever_arm = compute_lever_arm_squared(
            line_p1 = cas.vertcat(origin_x[3], origin_y[3]),   # origin_position
            line_p2=cas.vertcat(insertion_x[3], insertion_y[3]),  # insertion_position
            point = shoulder_position[:, i_state],  # shoulder_position
        )
        biceps_lever_arm_shoulder = compute_lever_arm_squared(
            line_p1 = cas.vertcat(origin_x[4], origin_y[4]),   # origin_position
            line_p2 = cas.vertcat(via_point_x[2], via_point_y[2]),   # via_point_position
            point = shoulder_position[:, i_state],  # shoulder_position
        )
        biceps_lever_arm_elbow = compute_lever_arm_squared(
            line_p1 = cas.vertcat(via_point_x[3], via_point_y[3]),   # via_point_position
            line_p2 = cas.vertcat(insertion_x[4], insertion_y[4]),  # insertion_position
            point = elbow_position[:, i_state],  # elbow_position
        )
        triceps_long_lever_arm_shoulder = compute_lever_arm_squared(
            line_p1 = cas.vertcat(origin_x[5], origin_y[5]),   # origin_position
            line_p2 = cas.vertcat(via_point_x[4], via_point_y[4]),   # via_point_position
            point = shoulder_position[:, i_state],  # shoulder_position
        )
        triceps_long_lever_arm_elbow = compute_lever_arm_squared(
            line_p1 = cas.vertcat(via_point_x[5], via_point_y[5]),   # via_point_position
            line_p2 = cas.vertcat(insertion_x[5], insertion_y[5]),  # insertion_position
            point = elbow_position[:, i_state],  # elbow_position
        )


        lever_arm_this_time = cas.vertcat(*[
                                            brachialis_lever_arm,
                                            triceps_lateral_lever_arm,
                                            deltoid_anterior_lever_arm,
                                            deltoid_posterior_lever_arm,
                                            biceps_lever_arm_shoulder, biceps_lever_arm_elbow,
                                            triceps_long_lever_arm_shoulder, triceps_long_lever_arm_elbow
                                          ],
        )
        objective += cas.sum1((lever_arm_this_time - lever_arm_to_match[:, i_state] ** 2) ** 2)


        # Match the muscle length
        i_muscle = 0
        for muscle_group in model.muscle_groups:
            for muscle in muscle_group.muscles:
                origin_rt = global_rt[muscle.origin_position.parent_name][i_state]
                muscle_origin_in_global = origin_rt @ muscle.origin_position.position
                insertion_rt = global_rt[muscle.insertion_position.parent_name][i_state]
                muscle_insertion_in_global = insertion_rt @ muscle.insertion_position.position
                opt_muscle_length = 0
                for i_via_point in range(nb_via_points):
                    if i_via_point == 0:
                        opt_muscle_length += (muscle_origin_in_global[:2] - cas.vertcat(via_point_x[i_via_point], via_point_y[i_via_point])) ** 2
                    else:
                        opt_muscle_length += (cas.vertcat(via_point_x[i_via_point - 1], via_point_y[i_via_point - 1]) - cas.vertcat(via_point_x[i_via_point], via_point_y[i_via_point])) ** 2
                opt_muscle_length += (muscle_insertion_in_global[:2] - cas.vertcat(via_point_x[-1], via_point_y[-1])) ** 2

                objective += cas.sum1((opt_muscle_length - muscle_length_to_match[i_muscle, i_state] ** 2) ** 2)
                i_muscle += 1



        w = cas.vertcat(via_point_x, via_point_y, origin_x, origin_y, insertion_x, insertion_y)
        w0 = np.zeros(w.shape)
        i_via_point = 0
        for muscle_group in model.muscle_groups:
            for muscle in muscle_group.muscles:
                for via_point in muscle.via_points:
                    via_point_rt = global_rt[via_point.parent_name][0]
                    via_point_in_global = via_point_rt @ via_point.position
                    w0[i_via_point, 0] = via_point_in_global[0, 0]
                    w0[i_via_point + nb_via_points, 0] = via_point_in_global[1, 0]
                    i_via_point += 1
        i_muscle = 0
        for muscle_group in model.muscle_groups:
            for muscle in muscle_group.muscles:
                muscle_rt = global_rt[muscle.origin_position.parent_name][0]
                muscle_in_global = muscle_rt @ muscle.origin_position.position
                w0[2* nb_via_points + i_muscle, 0] = muscle_in_global[0, 0]
                w0[2* nb_via_points + nb_muscles + i_muscle, 0] = muscle_in_global[1, 0]
                i_muscle += 1
        i_muscle = 0
        for muscle_group in model.muscle_groups:
            for muscle in muscle_group.muscles:
                muscle_rt = global_rt[muscle.insertion_position.parent_name][0]
                muscle_in_global = muscle_rt @ muscle.insertion_position.position
                w0[2* nb_via_points + 2* nb_muscles + i_muscle, 0] = muscle_in_global[0, 0]
                w0[2* nb_via_points + 3 * nb_muscles + i_muscle, 0] = muscle_in_global[1, 0]
                i_muscle += 1

        lbw = w0 - 0.2
        ubw = w0 + 0.2

        nlp = {'x': w, 'f': objective}
        solver = cas.nlpsol('solver', 'ipopt', nlp)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw)
        w_opt = sol['x']

    return

def create_initial_model() -> BiomechanicalModelReal:
    """
    Steps 1 + 2
    """

    # Create a model holder
    model = BiomechanicalModelReal()

    # Add segments
    model.add_segment(SegmentReal(name="base"))

    model.add_segment(
        SegmentReal(
            name="upper_arm",
            parent_name="base",
            rotations=Rotations.Z,
            dof_names=["shoulder_angle"],
            q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[0], max_bound=[np.pi / 2]),
            segment_coordinate_system=SegmentCoordinateSystemReal(scs=RotoTransMatrix(), is_scs_local=True),
            inertia_parameters=InertiaParametersReal(
                mass=1.4, center_of_mass=np.array([0.11, 0, 0]), inertia=np.array([0, 0, 0.025])
            ),
            mesh_file=MeshFileReal(
                mesh_file_name="mesh_cleaned/humerus.vtp",
                mesh_scale=None,
                mesh_rotation=np.array([0, 0, np.pi / 2]),
                mesh_translation=None,
            ),
        )
    )

    scs_lower_arm = RotoTransMatrix()
    scs_lower_arm.from_euler_angles_and_translation(
        angle_sequence="xyz",
        angles=np.array([0, 0, 0]),
        translation=np.array([0.3, 0, 0]),
    )
    model.add_segment(
        SegmentReal(
            name="lower_arm",
            parent_name="upper_arm",
            rotations=Rotations.Z,
            dof_names=["elbow_angle"],
            q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[0], max_bound=[np.pi]),
            segment_coordinate_system=SegmentCoordinateSystemReal(scs=scs_lower_arm, is_scs_local=True),
            inertia_parameters=InertiaParametersReal(
                mass=1.0, center_of_mass=np.array([0.16, 0, 0]), inertia=np.array([0, 0, 0.045])
            ),
            mesh_file=MeshFileReal(
                mesh_file_name="mesh_cleaned/radius.vtp",
                mesh_scale=None,
                mesh_rotation=np.array([0, 0 + 0.1, np.pi / 2 - 0.1]),
                mesh_translation=None,
            ),
        )
    )

    # Markers
    model.segments["lower_arm"].add_marker(
        MarkerReal(name="end_effector", parent_name="lower_arm", position=np.array([0.33, 0.0, 0]))
    )
    model.segments["base"].add_marker(
        MarkerReal(name="hand_start", parent_name="base", position=np.array([0.0, 0.2742, 0]))
    )
    model.segments["base"].add_marker(
        MarkerReal(name="hand_end", parent_name="base", position=np.array([0.0, 0.5273, 0]))
    )

    # Add muscles
    model.add_muscle_group(
        MuscleGroupReal(
            name="upper_arm_to_lower_arm",
            origin_parent_name="upper_arm",
            insertion_parent_name="lower_arm",
        ),
    )
    model.muscle_groups["upper_arm_to_lower_arm"].add_muscle(
        MuscleReal(
            name="brachialis",
            muscle_type=MuscleType.HILL_DE_GROOTE,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="upper_arm_to_lower_arm",
            origin_position=ViaPointReal(
                name="brachialis_origin",
                parent_name="upper_arm",
                muscle_name="brachialis",
                muscle_group="upper_arm_to_lower_arm",
                position=np.array([0.11, 0.025, 0.0]),  # Eye balled from the CoM!
            ),
            insertion_position=ViaPointReal(
                name="brachialis_insertion",
                parent_name="lower_arm",
                muscle_name="brachialis",
                muscle_group="upper_arm_to_lower_arm",
                position=np.array([0.025, 0.025, 0.0]),  # Eye balled !
            ),
            optimal_length=None,
            maximal_force=31.8 * 18,  # 31.8 N/cm2 * 18 cm2
            tendon_slack_length=None,
            pennation_angle=None,
            maximal_velocity=None,
            maximal_excitation=None,
        ),
    )
    model.muscle_groups["upper_arm_to_lower_arm"].muscles["brachialis"].add_via_point(
        ViaPointReal(
            name="brachialis_via_point",
            parent_name="upper_arm",
            muscle_name="brachialis",
            muscle_group="upper_arm_to_lower_arm",
            position=np.array([0.3, 0.025, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    model.muscle_groups["upper_arm_to_lower_arm"].add_muscle(
        MuscleReal(
            name="lateral_triceps",
            muscle_type=MuscleType.HILL_DE_GROOTE,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="upper_arm_to_lower_arm",
            origin_position=ViaPointReal(
                name="lateral_triceps_origin",
                parent_name="upper_arm",
                muscle_name="lateral_triceps",
                muscle_group="upper_arm_to_lower_arm",
                position=np.array([0.11, -0.025, 0.0]),  # Eye balled from the CoM!
            ),
            insertion_position=ViaPointReal(
                name="lateral_triceps_insertion",
                parent_name="lower_arm",
                muscle_name="lateral_triceps",
                muscle_group="upper_arm_to_lower_arm",
                position=np.array([0.025, -0.025, 0.0]),  # Eye balled !
            ),
            optimal_length=None,
            maximal_force=31.8 * 14,  # 31.8 N/cm2 * 14 cm2
            tendon_slack_length=None,
            pennation_angle=None,
            maximal_velocity=None,
            maximal_excitation=None,
        ),
    )
    model.muscle_groups["upper_arm_to_lower_arm"].muscles["lateral_triceps"].add_via_point(
        ViaPointReal(
            name="lateral_triceps_via_point",
            parent_name="upper_arm",
            muscle_name="lateral_triceps",
            muscle_group="upper_arm_to_lower_arm",
            position=np.array([0.3, -0.025, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    model.add_muscle_group(
        MuscleGroupReal(
            name="base_to_upper_arm",
            origin_parent_name="base",
            insertion_parent_name="upper_arm",
        ),
    )
    model.muscle_groups["base_to_upper_arm"].add_muscle(
        MuscleReal(
            name="anterior_deltoid",
            muscle_type=MuscleType.HILL_DE_GROOTE,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="base_to_upper_arm",
            origin_position=ViaPointReal(
                name="anterior_deltoid_origin",
                parent_name="base",
                muscle_name="anterior_deltoid",
                muscle_group="base_to_upper_arm",
                position=np.array([0.0, 0.07, 0.0]),  # Eye balled from a circle of r=0.05!
            ),
            insertion_position=ViaPointReal(
                name="anterior_deltoid_insertion",
                parent_name="upper_arm",
                muscle_name="anterior_deltoid",
                muscle_group="base_to_upper_arm",
                position=np.array([0.11, 0.025, 0.0]),  # Eye balled from the CoM!
            ),
            optimal_length=None,
            maximal_force=31.8 * 22,  # 31.8 N/cm2 * 22 cm2
            tendon_slack_length=None,
            pennation_angle=None,
            maximal_velocity=None,
            maximal_excitation=None,
        ),
    )
    model.muscle_groups["base_to_upper_arm"].add_muscle(
        MuscleReal(
            name="posterior_deltoid",
            muscle_type=MuscleType.HILL_DE_GROOTE,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="base_to_upper_arm",
            origin_position=ViaPointReal(
                name="posterior_deltoid_origin",
                parent_name="base",
                muscle_name="posterior_deltoid",
                muscle_group="base_to_upper_arm",
                position=np.array([0.0, -0.07, 0.0]),  # Eye balled from a circle of r=0.05!
            ),
            insertion_position=ViaPointReal(
                name="posterior_deltoid_insertion",
                parent_name="upper_arm",
                muscle_name="posterior_deltoid",
                muscle_group="base_to_upper_arm",
                position=np.array([0.11, -0.025, 0.0]),  # Eye balled from the CoM!
            ),
            optimal_length=None,
            maximal_force=31.8 * 12,  # 31.8 N/cm2 * 12 cm2
            tendon_slack_length=None,
            pennation_angle=None,
            maximal_velocity=None,
            maximal_excitation=None,
        ),
    )
    model.add_muscle_group(
        MuscleGroupReal(
            name="base_to_lower_arm",
            origin_parent_name="base",
            insertion_parent_name="lower_arm",
        ),
    )
    model.muscle_groups["base_to_lower_arm"].add_muscle(
        MuscleReal(
            name="biceps",
            muscle_type=MuscleType.HILL_DE_GROOTE,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="base_to_lower_arm",
            origin_position=ViaPointReal(
                name="biceps_origin",
                parent_name="base",
                muscle_name="biceps",
                muscle_group="base_to_lower_arm",
                position=np.array([-0.05, 0.05, 0.0]),  # Eye balled !
            ),
            insertion_position=ViaPointReal(
                name="biceps_insertion",
                parent_name="lower_arm",
                muscle_name="biceps",
                muscle_group="base_to_lower_arm",
                position=np.array([0.05, 0.025, 0.0]),  # Eye balled !
            ),
            optimal_length=None,
            maximal_force=31.8 * 5,  # 31.8 N/cm2 * 5 cm2
            tendon_slack_length=None,
            pennation_angle=None,
            maximal_velocity=None,
            maximal_excitation=None,
        ),
    )
    model.muscle_groups["base_to_lower_arm"].muscles["biceps"].add_via_point(
        ViaPointReal(
            name="biceps_shoulder_via_point",
            parent_name="upper_arm",
            muscle_name="biceps",
            muscle_group="base_to_lower_arm",
            position=np.array([0.0, 0.07, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    model.muscle_groups["base_to_lower_arm"].muscles["biceps"].add_via_point(
        ViaPointReal(
            name="biceps_elbow_via_point",
            parent_name="upper_arm",
            muscle_name="biceps",
            muscle_group="base_to_lower_arm",
            position=np.array([0.31, 0.025, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    model.muscle_groups["base_to_lower_arm"].add_muscle(
        MuscleReal(
            name="long_triceps",
            muscle_type=MuscleType.HILL_DE_GROOTE,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="base_to_lower_arm",
            origin_position=ViaPointReal(
                name="long_triceps_origin",
                parent_name="base",
                muscle_name="long_triceps",
                muscle_group="base_to_lower_arm",
                position=np.array([-0.05, -0.05, 0.0]),  # Eye balled !
            ),
            insertion_position=ViaPointReal(
                name="long_triceps_insertion",
                parent_name="lower_arm",
                muscle_name="long_triceps",
                muscle_group="base_to_lower_arm",
                position=np.array([0.05, -0.025, 0.0]),  # Eye balled !
            ),
            optimal_length=None,
            maximal_force=31.8 * 10,  # 31.8 N/cm2 * 10 cm2
            tendon_slack_length=None,
            pennation_angle=None,
            maximal_velocity=None,
            maximal_excitation=None,
        ),
    )
    model.muscle_groups["base_to_lower_arm"].muscles["long_triceps"].add_via_point(
        ViaPointReal(
            name="long_triceps_via_point",
            parent_name="upper_arm",
            muscle_name="long_triceps",
            muscle_group="base_to_lower_arm",
            position=np.array([0.0, -0.07, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    model.muscle_groups["base_to_lower_arm"].muscles["long_triceps"].add_via_point(
        ViaPointReal(
            name="long_triceps_via_point",
            parent_name="upper_arm",
            muscle_name="long_triceps",
            muscle_group="base_to_lower_arm",
            position=np.array([0.31, -0.025, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    return model



def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    visualization_flag = True

    # Paths
    biomod_filepath = "arm_model.bioMod"

    model = create_initial_model()
    via_point = find_via_point_that_matches_lever_arm(model)


if __name__ == "__main__":
    main()
