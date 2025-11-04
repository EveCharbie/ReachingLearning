"""
This file was used to create a 2D arm model from the 3D arm model of .

Originally, I wanted to match the lever arm from Li & Todorov 2004, but I think it is not geometrically possible.
So instead, the model was eye-balled, making sure the lever arm do not cross zero.
The muscle optimal length were set to the length at shoulder = 45° and elbow = 90° (we do not consider tendon).
"""

import logging
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
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
    FlatteningTool,
)

_logger = logging.getLogger(__name__)


def create_initial_model() -> BiomechanicalModelReal:

    # Paths
    current_path_file = Path(__file__).parent
    biomod_filepath = f"{current_path_file}/simple_arm_model.bioMod"
    osim_filepath = f"{current_path_file}/MOBL_ARMS_41.osim"
    geometry_path = f"{current_path_file}/mesh_cleaned"

    model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=geometry_path,
    )

    model.to_biomod(f"{current_path_file}/MOBL_ARMS_41.bioMod")
    model.animate()


if __name__ == "__main__":
    create_initial_model()
