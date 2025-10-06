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
import matplotlib.pyplot as plt

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

# Coefficients from Van Wouwe et al. 2022
#       a_shoul,  b_shoul,   c_shoul, a_elbow, b_elbow, c_elbow
dM_coefficients = np.array(
    [
        [0, 0, 0.001, 0.0300, -0.0110, 1.9000],  # brachialis = elbow flexor
        [0, 0, 0.001, -0.0190, 0, 0.001],  # lateral triceps = elbow extensor
        [0.0400, -0.0080, 1.9000, 0, 0, 0.001],  # anterior deltoid = shoulder flexor
        [-0.0420, 0, 0.001, 0, 0, 0.001],  # posterior deltoid = shoulder extensor
        [0.0300, -0.0110, 1.9000, 0.0320, -0.0100, 1.9000],  # biceps = shoulder and elbow flexor
        [-0.0390, 0, 0.001, -0.0220, 0, 0.001],  # long triceps = shoulder and elbow extensor
    ]
)


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
                position=np.array([0.11, 0.03, 0.0]),  # Eye balled from the CoM!
            ),
            insertion_position=ViaPointReal(
                name="brachialis_insertion",
                parent_name="lower_arm",
                muscle_name="brachialis",
                muscle_group="upper_arm_to_lower_arm",
                position=np.array([0.025, 0.015, 0.0]),  # Eye balled from the CoM!
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
            position=np.array([0.25, 0.025, 0.0]),  # Eye balled from a circle of r=0.05!
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
                position=np.array([0.035, -0.015, 0.0]),  # Eye balled from the CoM!
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
            name="lateral_triceps_via_point_1",
            parent_name="lower_arm",
            muscle_name="lateral_triceps",
            muscle_group="upper_arm_to_lower_arm",
            position=np.array([-0.025, -0.025, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    model.muscle_groups["upper_arm_to_lower_arm"].muscles["lateral_triceps"].add_via_point(
        ViaPointReal(
            name="lateral_triceps_via_point_2",
            parent_name="lower_arm",
            muscle_name="lateral_triceps",
            muscle_group="upper_arm_to_lower_arm",
            position=np.array([0.025, -0.025, 0.0]),  # Eye balled from a circle of r=0.05!
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
                position=np.array([-0.015, 0.04, 0.0]),  # Eye balled from a circle of r=0.05!
            ),
            insertion_position=ViaPointReal(
                name="anterior_deltoid_insertion",
                parent_name="upper_arm",
                muscle_name="anterior_deltoid",
                muscle_group="base_to_upper_arm",
                position=np.array([0.11, 0.015, 0.0]),  # Eye balled from the CoM!
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
                position=np.array([0.015, -0.04, 0.0]),  # Eye balled from a circle of r=0.05!
            ),
            insertion_position=ViaPointReal(
                name="posterior_deltoid_insertion",
                parent_name="upper_arm",
                muscle_name="posterior_deltoid",
                muscle_group="base_to_upper_arm",
                position=np.array([0.11, -0.015, 0.0]),  # Eye balled from the CoM!
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
                position=np.array([-0.04, 0.01, 0.0]),  # Eye balled !
            ),
            insertion_position=ViaPointReal(
                name="biceps_insertion",
                parent_name="lower_arm",
                muscle_name="biceps",
                muscle_group="base_to_lower_arm",
                position=np.array([0.05, 0.015, 0.0]),  # Eye balled from the CoM!
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
            parent_name="base",
            muscle_name="biceps",
            muscle_group="base_to_lower_arm",
            position=np.array([-0.03, 0.03, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    # model.muscle_groups["base_to_lower_arm"].muscles["biceps"].add_via_point(
    #     ViaPointReal(
    #         name="biceps_elbow_via_point",
    #         parent_name="upper_arm",
    #         muscle_name="biceps",
    #         muscle_group="base_to_lower_arm",
    #         position=np.array([0.28, 0.015, 0.0]),  # Eye balled from a circle of r=0.05!
    #     )
    # )
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
                position=np.array([-0.04, -0.03, 0.0]),  # Eye balled !
            ),
            insertion_position=ViaPointReal(
                name="long_triceps_insertion",
                parent_name="lower_arm",
                muscle_name="long_triceps",
                muscle_group="base_to_lower_arm",
                position=np.array([0.07, -0.015, 0.0]),  # Eye balled from the CoM!
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
            name="long_triceps_shoulder_via_point",
            parent_name="base",
            muscle_name="long_triceps",
            muscle_group="base_to_lower_arm",
            position=np.array([0.015, -0.04, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    model.muscle_groups["base_to_lower_arm"].muscles["long_triceps"].add_via_point(
        ViaPointReal(
            name="long_triceps_elbow_via_point_1",
            parent_name="lower_arm",
            muscle_name="long_triceps",
            muscle_group="base_to_lower_arm",
            position=np.array([-0.028, -0.015, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )
    model.muscle_groups["base_to_lower_arm"].muscles["long_triceps"].add_via_point(
        ViaPointReal(
            name="long_triceps_elbow_via_point_2",
            parent_name="lower_arm",
            muscle_name="long_triceps",
            muscle_group="base_to_lower_arm",
            position=np.array([0.028, -0.015, 0.0]),  # Eye balled from a circle of r=0.05!
        )
    )

    # model.animate()
    return model


def get_joint_position(model: BiomechanicalModelReal, q: np.ndarray, nb_states: int):
    # Get the RT in global for all n_states at once
    shoulder_position = np.zeros((2, nb_states))
    global_rt = model.forward_kinematics(q)
    elbow_position = np.zeros((2, nb_states))
    for i_state in range(nb_states):
        elbow_position[:, i_state] = global_rt["lower_arm"][i_state].translation[:2]
    return shoulder_position, elbow_position, global_rt


def get_compute_lever_arms_squared(
        brachialis_origin_global,
        lateral_triceps_origin_global,
        anterior_deltoid_origin_global,
        posterior_deltoid_origin_global,
        biceps_origin_global,
        long_triceps_origin_global,
        brachialis_insertion_global,
        lateral_triceps_insertion_global,
        anterior_deltoid_insertion_global,
        posterior_deltoid_insertion_global,
        biceps_insertion_global,
        long_triceps_insertion_global,
        brachialis_via_point_global,
        lateral_triceps_via_point_1_global,
        lateral_triceps_via_point_2_global,
        biceps_via_point_shoulder_global,
        biceps_via_point_elbow_global,
        long_triceps_via_point_shoulder_global,
        long_triceps_via_point_elbow_1_global,
        long_triceps_via_point_elbow_2_global,
        shoulder_position,
        elbow_position,
):
    # Match the lever_arm
    brachialis_lever_arm = compute_lever_arm_squared(
        line_p1=brachialis_via_point_global,
        line_p2=brachialis_insertion_global,
        point=elbow_position,
    )
    lateral_triceps_lever_arm = compute_lever_arm_squared(
        line_p1=lateral_triceps_via_point_1_global,
        line_p2=lateral_triceps_via_point_2_global,
        point=elbow_position,
    )
    anterior_deltoid_lever_arm = compute_lever_arm_squared(
        line_p1=anterior_deltoid_origin_global,
        line_p2=anterior_deltoid_insertion_global,
        point=shoulder_position,
    )
    posterior_deltoid_lever_arm = compute_lever_arm_squared(
        line_p1=posterior_deltoid_origin_global,
        line_p2=posterior_deltoid_insertion_global,
        point=shoulder_position,
    )
    biceps_lever_arm_shoulder = compute_lever_arm_squared(
        line_p1=biceps_origin_global,
        line_p2=biceps_via_point_shoulder_global,
        point=shoulder_position,
    )
    biceps_lever_arm_elbow = compute_lever_arm_squared(
        line_p1=biceps_via_point_elbow_global,
        line_p2=biceps_insertion_global,
        point=elbow_position,
    )
    long_triceps_lever_arm_shoulder = compute_lever_arm_squared(
        line_p1=long_triceps_origin_global,
        line_p2=long_triceps_via_point_shoulder_global,
        point=shoulder_position,
    )
    long_triceps_lever_arm_elbow = compute_lever_arm_squared(
        line_p1=long_triceps_via_point_elbow_1_global,
        line_p2=long_triceps_via_point_elbow_2_global,
        point=elbow_position,
    )

    return (
        brachialis_lever_arm,
        lateral_triceps_lever_arm,
        anterior_deltoid_lever_arm,
        posterior_deltoid_lever_arm,
        biceps_lever_arm_shoulder,
        biceps_lever_arm_elbow,
        long_triceps_lever_arm_shoulder,
        long_triceps_lever_arm_elbow,
    )

def find_via_point_that_matches_lever_arm(model: BiomechanicalModelReal, q: np.ndarray) -> np.ndarray:
    """
    Step 3
    """
    nb_states = q.shape[1]

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

    shoulder_position, elbow_position, global_rt = get_joint_position(model, q, nb_states)

    # Initialize problem
    nb_via_points = model.nb_via_points
    nb_muscles = model.nb_muscles
    via_point_x = cas.MX.sym("via_point_x", nb_via_points, 1)
    via_point_y = cas.MX.sym("via_point_y", nb_via_points, 1)
    origin_x = cas.MX.sym("origin_x", 6, 1)
    origin_y = cas.MX.sym("origin_y", 6, 1)
    insertion_x = cas.MX.sym("insertion_x", 6, 1)
    insertion_y = cas.MX.sym("insertion_y", 6, 1)
    j = 0
    g = []
    lbg = []
    ubg = []

    global_rt = model.forward_kinematics(q)

    # Test joint configurations
    for i_state in range(nb_states):

        # Origins
        brachialis_origin_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(origin_x[0], origin_y[0])
        lateral_triceps_origin_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(origin_x[1], origin_y[1])
        anterior_deltoid_origin_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ cas.vertcat(origin_x[2], origin_y[2])
        posterior_deltoid_origin_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ cas.vertcat(origin_x[3], origin_y[3])
        biceps_origin_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ cas.vertcat(origin_x[4], origin_y[4])
        long_triceps_origin_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ cas.vertcat(origin_x[5], origin_y[5])

        # Insertions
        brachialis_insertion_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(insertion_x[0], insertion_y[0])
        lateral_triceps_insertion_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(insertion_x[1], insertion_y[1])
        anterior_deltoid_insertion_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(insertion_x[2], insertion_y[2])
        posterior_deltoid_insertion_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(insertion_x[3], insertion_y[3])
        biceps_insertion_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(insertion_x[4], insertion_y[4])
        long_triceps_insertion_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(insertion_x[5], insertion_y[5])

        # Via points
        brachialis_via_point_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(via_point_x[0], via_point_y[0])
        lateral_triceps_via_point_1_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(via_point_x[1], via_point_y[1])
        lateral_triceps_via_point_2_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(via_point_x[2], via_point_y[2])
        biceps_via_point_shoulder_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(via_point_x[3], via_point_y[3])
        biceps_via_point_elbow_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(via_point_x[4], via_point_y[4])
        long_triceps_via_point_shoulder_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ cas.vertcat(via_point_x[5], via_point_y[5])
        long_triceps_via_point_elbow_1_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(via_point_x[6], via_point_y[6])
        long_triceps_via_point_elbow_2_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ cas.vertcat(via_point_x[7], via_point_y[7])

        (
            brachialis_lever_arm,
            lateral_triceps_lever_arm,
            anterior_deltoid_lever_arm,
            posterior_deltoid_lever_arm,
            biceps_lever_arm_shoulder,
            biceps_lever_arm_elbow,
            long_triceps_lever_arm_shoulder,
            long_triceps_lever_arm_elbow,
        ) = get_compute_lever_arms_squared(
            brachialis_origin_global,
            lateral_triceps_origin_global,
            anterior_deltoid_origin_global,
            posterior_deltoid_origin_global,
            biceps_origin_global,
            long_triceps_origin_global,
            brachialis_insertion_global,
            lateral_triceps_insertion_global,
            anterior_deltoid_insertion_global,
            posterior_deltoid_insertion_global,
            biceps_insertion_global,
            long_triceps_insertion_global,
            brachialis_via_point_global,
            lateral_triceps_via_point_1_global,
            lateral_triceps_via_point_2_global,
            biceps_via_point_shoulder_global,
            biceps_via_point_elbow_global,
            long_triceps_via_point_shoulder_global,
            long_triceps_via_point_elbow_1_global,
            long_triceps_via_point_elbow_2_global,
            shoulder_position[:, i_state],
            elbow_position[:, i_state],
        )

        lever_arm_this_time = cas.vertcat(*[
                                            brachialis_lever_arm,
                                            lateral_triceps_lever_arm,
                                            anterior_deltoid_lever_arm,
                                            posterior_deltoid_lever_arm,
                                            biceps_lever_arm_shoulder, biceps_lever_arm_elbow,
                                            long_triceps_lever_arm_shoulder, long_triceps_lever_arm_elbow
                                          ],
        )
        j += cas.sum1((lever_arm_this_time - lever_arm_to_match[:, i_state] ** 2) ** 2)


        # Match the muscle length
        # Brachialis
        i_muscle = 0
        opt_muscle_length_brachialis = (
            ((brachialis_insertion_global - brachialis_via_point_global) +
            (brachialis_via_point_global - brachialis_origin_global))
             ) ** 2
        # j += cas.sum1((opt_muscle_length_brachialis - muscle_length_to_match[i_muscle, i_state]) ** 2)
        # g += [opt_muscle_length_brachialis - muscle_length_to_match[i_muscle, i_state]]
        # lbg += [-0.05]
        # ubg += [0.05]
        j += 0.01 * cas.sum1(opt_muscle_length_brachialis)

        # Lateral triceps
        i_muscle = 1
        opt_muscle_length_lateral_triceps = (
            ((lateral_triceps_insertion_global - lateral_triceps_via_point_2_global) +
             (lateral_triceps_via_point_2_global - lateral_triceps_via_point_1_global) +
            (lateral_triceps_via_point_1_global - lateral_triceps_origin_global))
             ) ** 2
        # j += cas.sum1((opt_muscle_length_lateral_triceps - muscle_length_to_match[i_muscle, i_state]) ** 2 )
        # g += [opt_muscle_length_lateral_triceps - muscle_length_to_match[i_muscle, i_state]]
        # lbg += [-0.05]
        # ubg += [0.05]
        j += 0.01 * cas.sum1(opt_muscle_length_lateral_triceps)

        # Anterior deltoid
        i_muscle = 2
        opt_muscle_length_anterior_deltoid = (
            (anterior_deltoid_insertion_global- anterior_deltoid_origin_global)
             ) ** 2
        # j += cas.sum1((opt_muscle_length_anterior_deltoid - muscle_length_to_match[i_muscle, i_state]) **2)
        # g += [opt_muscle_length_anterior_deltoid - muscle_length_to_match[i_muscle, i_state]]
        # lbg += [-0.05]
        # ubg += [0.05]
        j += 0.01 * cas.sum1(opt_muscle_length_anterior_deltoid)

        # Posterior deltoid
        i_muscle = 3
        opt_muscle_length_posterior_deltoid = (
            (posterior_deltoid_insertion_global- posterior_deltoid_origin_global)
             ) ** 2
        # j += cas.sum1((opt_muscle_length_posterior_deltoid - muscle_length_to_match[i_muscle, i_state]) ** 2)
        # g += [opt_muscle_length_posterior_deltoid - muscle_length_to_match[i_muscle, i_state]]
        # lbg += [-0.05]
        # ubg += [0.05]
        j += 0.1 * cas.sum1(opt_muscle_length_posterior_deltoid)

        # Biceps
        i_muscle = 4
        opt_muscle_length_biceps = (
            (biceps_insertion_global - biceps_via_point_elbow_global) +
            (biceps_via_point_elbow_global - biceps_via_point_shoulder_global) +
            (biceps_via_point_shoulder_global - biceps_origin_global)
             ) ** 2
        # j += cas.sum1((opt_muscle_length_biceps - muscle_length_to_match[i_muscle, i_state]) ** 2)
        # g += [opt_muscle_length_biceps - muscle_length_to_match[i_muscle, i_state]]
        # lbg += [-0.05]
        # ubg += [0.05]
        j += 0.01 * cas.sum1(opt_muscle_length_biceps)

        # Biceps
        i_muscle = 5
        opt_muscle_length_long_triceps = (
            (long_triceps_insertion_global - long_triceps_via_point_elbow_2_global) +
            (long_triceps_via_point_elbow_2_global - long_triceps_via_point_elbow_1_global) +
            (long_triceps_via_point_elbow_1_global - long_triceps_via_point_shoulder_global) +
            (long_triceps_via_point_shoulder_global - long_triceps_origin_global)
        ) ** 2
        # j += cas.sum1((opt_muscle_length_long_triceps - muscle_length_to_match[i_muscle, i_state]) ** 2)
        # g += [opt_muscle_length_long_triceps - muscle_length_to_match[i_muscle, i_state]]
        # lbg += [-0.05]
        # ubg += [0.05]
        j += 0.01 * cas.sum1(opt_muscle_length_long_triceps)

    w = cas.vertcat(via_point_x, via_point_y, origin_x, origin_y, insertion_x, insertion_y)
    w0 = np.zeros(w.shape)
    i_via_point = 0
    for muscle_group in model.muscle_groups:
        for muscle in muscle_group.muscles:
            for via_point in muscle.via_points:
                w0[i_via_point, 0] = via_point.position[0, 0]
                w0[i_via_point + nb_via_points, 0] = via_point.position[1, 0]
                i_via_point += 1
    i_muscle = 0
    for muscle_group in model.muscle_groups:
        for muscle in muscle_group.muscles:
            w0[2* nb_via_points + i_muscle, 0] = muscle.origin_position.position[0, 0]
            w0[2* nb_via_points + nb_muscles + i_muscle, 0] = muscle.origin_position.position[1, 0]
            i_muscle += 1
    i_muscle = 0
    for muscle_group in model.muscle_groups:
        for muscle in muscle_group.muscles:
            w0[2* nb_via_points + 2* nb_muscles + i_muscle, 0] = muscle.insertion_position.position[0, 0]
            w0[2* nb_via_points + 3 * nb_muscles + i_muscle, 0] = muscle.insertion_position.position[1, 0]
            i_muscle += 1

    lbw = w0 - 0.05
    ubw = w0 + 0.05

    nlp = {'x': w, 'f': j}  #, 'g': cas.vertcat(*g)}
    opts = {"ipopt.max_iter": 0}
    solver = cas.nlpsol('solver', 'ipopt', nlp, opts)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw)  #, lbg=lbg, ubg=ubg)
    w_opt = sol['x']

    return w_opt


def set_via_points_from_optimization(model: BiomechanicalModelReal, w_opt: np.ndarray) -> BiomechanicalModelReal:
    """
    Step 4
    """
    nb_via_points = model.nb_via_points
    nb_muscles = model.nb_muscles
    i_via_point = 0
    for muscle_group in model.muscle_groups:
        for muscle in muscle_group.muscles:
            for via_point in muscle.via_points:
                via_point.position[0] = w_opt[i_via_point, 0]
                via_point.position[1] = w_opt[i_via_point + nb_via_points, 0]
                i_via_point += 1
    i_muscle = 0
    for muscle_group in model.muscle_groups:
        for muscle in muscle_group.muscles:
            muscle.origin_position.position[0] = w_opt[2* nb_via_points + i_muscle, 0]
            muscle.origin_position.position[1] = w_opt[2* nb_via_points + nb_muscles + i_muscle, 0]
            muscle.insertion_position.position[0] = w_opt[2* nb_via_points + 2* nb_muscles + i_muscle, 0]
            muscle.insertion_position.position[1] = w_opt[2* nb_via_points + 3 * nb_muscles + i_muscle, 0]
            i_muscle += 1
    return model

def plot_lever_arms(model: BiomechanicalModelReal, q: np.ndarray):
    """
    Plot the lever arms to verify the fitting
    """
    nb_states = q.shape[1]

    shoulder_lever_arms = np.zeros((6, nb_states))
    elbow_lever_arms = np.zeros((6, nb_states))
    for i_state in range(nb_states):
        for i_muscle in range(model.nb_muscles):
            shoulder_lever_arms[i_muscle, i_state] = lever_arm_estimation(dM_coefficients[i_muscle, 0], dM_coefficients[i_muscle, 1], dM_coefficients[i_muscle, 2], q[0, i_state])
            elbow_lever_arms[i_muscle, i_state] = lever_arm_estimation(dM_coefficients[i_muscle, 3], dM_coefficients[i_muscle, 4], dM_coefficients[i_muscle, 5], q[1, i_state])

    colors = ["tab:purple", "k", "y", "tab:green", "tab:pink", "tab:blue"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for i_muscle in range(model.nb_muscles):
        axs[0].plot(q[0, :] * 180 / np.pi, np.abs(shoulder_lever_arms[i_muscle, :]), label=model.muscle_names[i_muscle], color=colors[i_muscle])
        axs[1].plot(q[1, :] * 180 / np.pi, np.abs(elbow_lever_arms[i_muscle, :]), label=model.muscle_names[i_muscle], color=colors[i_muscle])


    # Plot optimal lever arms
    shoulder_position, elbow_position, global_rt = get_joint_position(model, q, nb_states)

    shoulder_lever_arms_opt = np.zeros((6, nb_states))
    elbow_lever_arms_opt = np.zeros((6, nb_states))
    for i_state in range(nb_states):

        # Origins
        brachialis_origin_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["upper_arm_to_lower_arm"].muscles["brachialis"].origin_position.position[:2, 0]
        lateral_triceps_origin_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["upper_arm_to_lower_arm"].muscles["lateral_triceps"].origin_position.position[:2, 0]
        anterior_deltoid_origin_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_upper_arm"].muscles["anterior_deltoid"].origin_position.position[:2, 0]
        posterior_deltoid_origin_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_upper_arm"].muscles["posterior_deltoid"].origin_position.position[:2, 0]
        biceps_origin_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_lower_arm"].muscles["biceps"].origin_position.position[:2, 0]
        long_triceps_origin_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_lower_arm"].muscles["long_triceps"].origin_position.position[:2, 0]

        # Insertions
        brachialis_insertion_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["upper_arm_to_lower_arm"].muscles["brachialis"].insertion_position.position[:2, 0]
        lateral_triceps_insertion_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["upper_arm_to_lower_arm"].muscles["lateral_triceps"].insertion_position.position[:2, 0]
        anterior_deltoid_insertion_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_upper_arm"].muscles["anterior_deltoid"].insertion_position.position[:2, 0]
        posterior_deltoid_insertion_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_upper_arm"].muscles["posterior_deltoid"].insertion_position.position[:2, 0]
        biceps_insertion_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_lower_arm"].muscles["biceps"].insertion_position.position[:2, 0]
        long_triceps_insertion_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_lower_arm"].muscles["long_triceps"].insertion_position.position[:2, 0]

        # Via points
        brachialis_via_point_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["upper_arm_to_lower_arm"].muscles["brachialis"].via_points[0].position[:2, 0]
        lateral_triceps_via_point_1_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["upper_arm_to_lower_arm"].muscles["lateral_triceps"].via_points[0].position[:2, 0]
        lateral_triceps_via_point_2_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["upper_arm_to_lower_arm"].muscles["lateral_triceps"].via_points[1].position[:2, 0]
        biceps_via_point_shoulder_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_lower_arm"].muscles["biceps"].via_points[0].position[:2, 0]
        biceps_via_point_elbow_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_lower_arm"].muscles["biceps"].via_points[1].position[:2, 0]
        long_triceps_via_point_shoulder_global = global_rt["base"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_lower_arm"].muscles["long_triceps"].via_points[0].position[:2, 0]
        long_triceps_via_point_elbow_1_global = global_rt["upper_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_lower_arm"].muscles["long_triceps"].via_points[1].position[:2, 0]
        long_triceps_via_point_elbow_2_global = global_rt["lower_arm"][i_state].rt_matrix[:2, :2] @ model.muscle_groups["base_to_lower_arm"].muscles["long_triceps"].via_points[2].position[:2, 0]

        (
            brachialis_lever_arm,
            lateral_triceps_lever_arm,
            anterior_deltoid_lever_arm,
            posterior_deltoid_lever_arm,
            biceps_lever_arm_shoulder,
            biceps_lever_arm_elbow,
            long_triceps_lever_arm_shoulder,
            long_triceps_lever_arm_elbow,
        ) = get_compute_lever_arms_squared(
            brachialis_origin_global,
            lateral_triceps_origin_global,
            anterior_deltoid_origin_global,
            posterior_deltoid_origin_global,
            biceps_origin_global,
            long_triceps_origin_global,
            brachialis_insertion_global,
            lateral_triceps_insertion_global,
            anterior_deltoid_insertion_global,
            posterior_deltoid_insertion_global,
            biceps_insertion_global,
            long_triceps_insertion_global,
            brachialis_via_point_global,
            lateral_triceps_via_point_1_global,
            lateral_triceps_via_point_2_global,
            biceps_via_point_shoulder_global,
            biceps_via_point_elbow_global,
            long_triceps_via_point_shoulder_global,
            long_triceps_via_point_elbow_1_global,
            long_triceps_via_point_elbow_2_global,
            shoulder_position[:, i_state],
            elbow_position[:, i_state],
        )

        shoulder_lever_arms_opt[:, i_state] = np.array([
            0.0,  # brachialis has no shoulder lever arm
            0.0,  # lateral triceps has no shoulder lever arm
            np.sqrt(float(anterior_deltoid_lever_arm)),
            np.sqrt(float(posterior_deltoid_lever_arm)),
            np.sqrt(float(biceps_lever_arm_shoulder)),
            np.sqrt(float(long_triceps_lever_arm_shoulder))
        ])
        elbow_lever_arms_opt[:, i_state] = np.array([
            np.sqrt(float(brachialis_lever_arm)),
            np.sqrt(float(lateral_triceps_lever_arm)),
            0.0,  # anterior deltoid has no elbow lever arm
            0.0,  # posterior deltoid has no elbow lever arm
            np.sqrt(float(biceps_lever_arm_elbow)),
            np.sqrt(float(long_triceps_lever_arm_elbow))
        ])

    for i_muscle in range(model.nb_muscles):
        axs[0].plot(q[0, :] * 180 / np.pi, np.abs(shoulder_lever_arms_opt[i_muscle, :]), '--', color=colors[i_muscle])
        axs[0].plot(q[0, :] * 180 / np.pi, np.abs(shoulder_lever_arms_opt[i_muscle, :]), ':k')
        axs[1].plot(q[1, :] * 180 / np.pi, np.abs(elbow_lever_arms_opt[i_muscle, :]), '--', color=colors[i_muscle])
        axs[1].plot(q[1, :] * 180 / np.pi, np.abs(elbow_lever_arms_opt[i_muscle, :]), ':k')

    axs[0].set_title("Shoulder lever arms")
    axs[0].set_xlabel("Shoulder angle [deg]")
    axs[0].set_ylabel("Lever arm [m]")
    axs[1].set_title("Elbow lever arms")
    axs[1].set_xlabel("Elbow angle [deg]")
    axs[1].set_ylabel("Lever arm [m]")
    axs[1].legend()

    plt.savefig("optimal_lever_arms.png")
    plt.show()


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

    # Create tht initial model
    model = create_initial_model()

    # Get the states to consider for the optimization
    nb_states = 100
    q = np.zeros((model.nb_q, nb_states))
    q[0, :] = np.linspace(0, np.pi/2, nb_states)
    q[1, :] = np.linspace(0, np.pi, nb_states)

    # # Optimize the position of the muscle via points, origins and insertions
    # w_opt = find_via_point_that_matches_lever_arm(model, q)
    #
    # # Update the model with the optimized via points
    # model = set_via_points_from_optimization(model, w_opt)
    #
    # # Plot the results
    # plot_lever_arms(model, q)

    # Save the model
    model.to_biomod(biomod_filepath)

    if visualization_flag:
        import pyorerun

        animation = pyorerun.LiveModelAnimation(biomod_filepath, with_q_charts=True)
        animation.rerun()



if __name__ == "__main__":
    main()
