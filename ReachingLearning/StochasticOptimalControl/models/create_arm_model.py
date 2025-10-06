"""
This file was used to create the arm model used in the reaching learning example.

Originally, I wanted to match the lever arm from Li & Todorov 2004, but I think it is not geometrically possible.
So instead, the model was eye-balled, making sure the lever arm do not cross zero.
The muscle optimal length were set to the length at shoulder = 45° and elbow = 90°.
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


def create_initial_model() -> BiomechanicalModelReal:

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

    # Save the model
    model.to_biomod(biomod_filepath)

    if visualization_flag:
        import pyorerun

        animation = pyorerun.LiveModelAnimation(biomod_filepath, with_q_charts=True)
        animation.rerun()



if __name__ == "__main__":
    main()
