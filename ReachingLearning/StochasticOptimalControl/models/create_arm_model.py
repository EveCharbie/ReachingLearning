"""
This file was used to create the arm model used in the reaching learning example.
"""

import logging
from pathlib import Path
import numpy as np

from biobuddy import (
    MuscleType,
    MuscleStateType,
    MeshParser,
    MeshFormat,
    BiomechanicalModelReal,
    Rotations,
    Translations,
    RotoTransMatrix,
    SymmetryTool,
)

_logger = logging.getLogger(__name__)


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
    )
    visualization_flag = True

    # Paths
    current_path_file = Path(__file__).parent
    biomod_filepath = "MOBL_ARMS_41.bioMod"
    osim_filepath = "MOBL_ARMS_41.osim"
    geometry_path = "mesh_cleaned"

    # # Convert the vtp files
    # mesh = MeshParser(geometry_folder="mesh")
    # mesh.process_meshes(fail_on_error=False)
    # mesh.write(geometry_path, format=MeshFormat.VTP)

    # Read the .osim file
    model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=geometry_path,
    )
    # Fix the via points before translating to biomod as there are some conditional and moving via points
    model.fix_via_points(q=np.zeros((model.nb_q,)))

    # Removing unused segments
    segments_to_remove = [
                        'thorax_parent_offset',
                        'thorax_translation',
                        'thorax_rotation_transform',
                        'thorax',
                        'clavicle_parent_offset',
                        'clavicle_translation',
                        'thorax_offset_sternoclavicular_r2',
                        'thorax_offset_sternoclavicular_r3',
                        'clavicle_rotation_2',
                        'clavicle_reset_axis',
                        'clavicle',
                        'clavphant_parent_offset',
                        'clavphant_translation',
                        'clavicle_offset_unrotscap_r3',
                        'clavicle_offset_unrotscap_r2',
                        'clavphant_rotation_2',
                        'clavphant_reset_axis',
                        'clavphant',
                        'proximal_row_parent_offset',
                         'proximal_row_translation',
                         'radius_offset_deviation',
                         'proximal_row_rotation_1',
                         'radius_offset_flexion',
                         'proximal_row_reset_axis',
                         'proximal_row_geom_2',
                         'proximal_row_geom_3',
                         'proximal_row_geom_4',
                         'proximal_row',
                         'hand_parent_offset',
                         'hand_translation',
                         'proximal_row_offset_wrist_hand_r1',
                         'proximal_row_offset_wrist_hand_r3',
                         'hand_rotation_2',
                         'hand_reset_axis',
                         'hand_geom_2',
                         'hand_geom_3',
                         'hand_geom_4',
                         'hand_geom_5',
                         'hand_geom_6',
                         'hand_geom_7',
                         'hand_geom_8',
                         'hand_geom_9',
                         'hand_geom_10',
                         'hand_geom_11',
                         'hand_geom_12',
                         'hand_geom_13',
                         'hand_geom_14',
                         'hand_geom_15',
                         'hand_geom_16',
                         'hand_geom_17',
                         'hand_geom_18',
                         'hand_geom_19',
                         'hand_geom_20',
                         'hand_geom_21',
                         'hand_geom_22',
                         'hand_geom_23',
                         'hand']
    for segment in segments_to_remove:
        if segment in model.segment_names:
            model.remove_segment(segment)
    model.segments["scapula_parent_offset"].parent_name = "ground"
    # model.segments["humphant_rotation_1"].parent_name = "ground"

    # Removing unused muscles
    muscles_to_remove = [
        'DELT1',
        'PECM1',
        'DELT2',
        'SUPSP',
        'INFSP',
        'SUBSC',
        'TMIN',
        'TMAJ',
        'DELT3',
        'CORB',
        'PECM2',
        'PECM3',
        'LAT1',
        'LAT2',
        'LAT3',
        'ANC',
        'SUP',
        'PQ',
        'BRD',
        'PT',
        'ECRL',
        'ECRB',
        'ECU',
        'FCR',
        'FCU',
        'PL',
        'FDSL',
        'FDSR',
        'EDCL',
        'EDCR',
        'EDCM',
        'EDCI',
        'EDM',
        'FDSM',
        'FDSI',
        'FDPL',
        'FDPR',
        'FDPM',
        'FDPI',
        'EIP',
        'EPL',
        'EPB',
        'FPL',
        'APL',
    ]
    for muscle_group in  model.muscle_groups.copy():
        for muscle in muscle_group.muscles.copy():
            if muscle.name in muscles_to_remove:
                model.muscle_groups[muscle_group.name].remove_muscle(muscle.name)

    model.update_muscle_groups()

    # Remove markers
    for segment in model.segments:
        segment.markers = []

    # Remove rotations between the scapula and the humerus frames
    segment_to_remove_rotation =  [
        "clavphant_offset_acromioclavicular_r2",
        "clavphant_offset_acromioclavicular_r3",
        "clavphant_offset_acromioclavicular_r1",
        "scapula_reset_axis",
    ]
    for segment_name in segment_to_remove_rotation:
        model.segments[segment_name].segment_coordinate_system.scs = RotoTransMatrix()

    # Place the arm in T-pose
    evelation_idx = model.dof_names.index('humphant_offset_shoulder_elv')
    q_static = np.zeros((model.nb_q,))
    q_static[evelation_idx] = np.pi / 2
    model.modify_model_static_pose(q_static)

    # Removing unused degrees of freedom
    dofs_to_remove = [
         'ground_offset_t_x',
         'ground_offset_t_y',
         'ground_offset_t_z',
         'ground_offset_r_x',
         'ground_offset_r_y',
         'ground_offset_r_z',
         'thorax_offset_sternoclavicular_r2',
         'thorax_offset_sternoclavicular_r3',
         'clavicle_offset_unrotscap_r3',
         'clavicle_offset_unrotscap_r2',
         'clavphant_offset_acromioclavicular_r2',
         'clavphant_offset_acromioclavicular_r3',
         'clavphant_offset_acromioclavicular_r1',
         'scapula_offset_unrothum_r1',
         'scapula_offset_unrothum_r3',
         'scapula_offset_unrothum_r2',
         'scapphant_offset_elv_angle',
         'humphant_offset_shoulder_elv',
         'humphant_offset_shoulder1_r2',
         'humphant1_offset_shoulder_rot',
         'ulna_offset_pro_sup',
         'radius_offset_deviation',
         'radius_offset_flexion'
    ]
    for dof_name in dofs_to_remove:
        for segment in model.segments:
            if dof_name in segment.dof_names:
                segment.remove_dof(dof_name)

    # # Add a shoulder dof in the right joint coordinate system
    # model.segments["humphant_offset_shoulder_elv"].rotations = Rotations.Z
    # model.segments["humphant_offset_shoulder_elv"].dof_names = ["shoulder_rotZ"]

    # Remove more segments
    for segment_name in model.get_chain_between_segments("scapula_translation", "scapula")[:-1]:
        model.remove_segment(segment_name)
    model.segments["scapula"].parent_name = "scapula_parent_offset"
    model.segments["scapphant_parent_offset"].segment_coordinate_system.scs.translation = np.array([0, 0, 0])
    model.segments["scapula"].mesh_file = None

    for segment_name in model.get_chain_between_segments("scapphant_translation", "scapphant").copy()[:-1]:
        model.remove_segment(segment_name)
    model.segments["scapphant"].parent_name = "scapphant_parent_offset"

    for segment_name in model.get_chain_between_segments("humphant_translation", "humphant").copy()[:-1]:
        model.remove_segment(segment_name)
    model.segments["humphant"].parent_name = "humphant_parent_offset"

    for segment_name in model.get_chain_between_segments("humphant1_translation", "humphant1").copy()[:-1]:
        model.remove_segment(segment_name)
    model.segments["humphant1"].parent_name = "humphant1_parent_offset"
    model.segments["humphant1"].rotations = Rotations.Z
    model.segments["humphant1"].dof_names = ["shoulder_rotZ"]

    for segment_name in model.get_chain_between_segments("humerus_translation", "humerus").copy()[:-1]:
        model.remove_segment(segment_name)
    model.segments["humerus"].parent_name = "humerus_parent_offset"

    for segment_name in model.get_chain_between_segments("ulna_translation", "ulna").copy()[:-1]:
        model.remove_segment(segment_name)
    model.segments["ulna"].parent_name = "ulna_parent_offset"
    model.segments["ulna"].rotations = Rotations.Z
    model.segments["ulna"].dof_names = ["elbow_rotZ"]

    for segment_name in model.get_chain_between_segments("radius_translation", "radius").copy()[:-1]:
        model.remove_segment(segment_name)
    model.segments["radius"].parent_name = "radius_parent_offset"


    # Place the zero at the shoulder center
    global_jcs = model.forward_kinematics()
    shoulder_position = global_jcs["humerus"][0].translation
    model.segments["ground"].segment_coordinate_system.scs.translation -= shoulder_position


    # Symmetrize the model
    symmetry_tool = SymmetryTool(model, axis=Translations.Z)
    model = symmetry_tool.symmetrize()


    # And convert it to a .bioMod file
    model.to_biomod(biomod_filepath, with_mesh=visualization_flag)

    if visualization_flag:
        import pyorerun

        animation = pyorerun.LiveModelAnimation(biomod_filepath, with_q_charts=True)
        animation.rerun()



if __name__ == "__main__":
    main()
