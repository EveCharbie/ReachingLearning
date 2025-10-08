import numpy as np
import pyorerun


def animate_socp_basic(
    final_time,
    n_shooting,
    q_opt,
    muscles_opt,
):

    # Add the model
    model = pyorerun.BiorbdModel("../ReachingLearning/StochasticOptimalControl/models/arm_model.bioMod")
    model.options.show_marker_labels = False
    model.options.show_center_of_mass_labels = False
    model.options.show_muscle_labels = False

    # Add the end effector as persistent marker
    model.options.persistent_markers = pyorerun.PersistentMarkerOptions(
        marker_names=["end_effector"],
        radius=0.005,
        color=np.array([0, 1, 0]),
        show_labels=False,
        nb_frames=n_shooting + 1,
    )

    # Initialize the animation
    t_span = np.linspace(0, final_time, n_shooting + 1)
    viz = pyorerun.PhaseRerun(t_span)

    # Add experimental emg
    pyoemg = pyorerun.PyoMuscles(
        data=np.hstack((muscles_opt, np.zeros((6, 1)))),
        muscle_names=list(model.muscle_names),
        mvc=np.ones((model.nb_muscles,)),
        colormap="viridis",
    )

    # Add the kinematics
    viz.add_animated_model(model, q_opt, muscle_activations_intensity=pyoemg)

    # Play
    viz.rerun("OCP solution")
