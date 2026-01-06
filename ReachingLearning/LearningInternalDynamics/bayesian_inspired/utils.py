from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg'
import casadi as cas
import biorbd_casadi as biorbd

from ...StochasticOptimalControl.utils import RK4, get_dm_value


def get_the_real_dynamics():
    current_path = Path(__file__).parent
    model_path = f"{current_path}/../../StochasticOptimalControl/models/arm_model.bioMod"
    biorbd_model = biorbd.Biorbd(model_path)
    nb_q = biorbd_model.nb_q
    X = cas.MX.sym("x", nb_q * 2)
    U = cas.MX.sym("u", nb_q)
    motor_noise = cas.MX.sym("motor_noise", nb_q)

    xdot = cas.vertcat(X[nb_q:], biorbd_model.forward_dynamics(X[:nb_q], X[nb_q:], U))
    real_dynamics = cas.Function("forward_dynamics", [X, U, motor_noise], [xdot])

    inv_mass_matrix_func = cas.Function(
        "inv_mass_matrix",
        [X],
        [cas.inv(biorbd_model.mass_matrix(X[:nb_q]))],
    )
    nl_effect_vector_func = cas.Function(
        "nl_effect_vector",
        [X],
        [biorbd_model.non_linear_effect(X[:nb_q], X[nb_q:])],
    )

    return real_dynamics, inv_mass_matrix_func, nl_effect_vector_func

def get_the_real_marker_position():
    current_path = Path(__file__).parent
    model_path = f"{current_path}/../../StochasticOptimalControl/models/arm_model.bioMod"
    biorbd_model = biorbd.Model(model_path)
    nb_q = biorbd_model.nbQ()
    Q = cas.MX.sym("q", nb_q)

    marker_index = biorbd.marker_index(biorbd_model, "end_effector")
    real_marker_func = cas.Function("marker", [Q], [biorbd_model.marker(Q, marker_index).to_mx()[:2]])

    return real_marker_func

def integrate_the_dynamics(
        x0: np.ndarray,
        u: np.ndarray,
        dt: float,
        current_forward_dyn,
        real_forward_dyn: cas.Function,
        inv_mass_matrix_func: cas.Function,
        nl_effect_vector_func: cas.Function,
):
    nb_q = 2
    n_shooting = u.shape[1]
    x_integrated_approx = np.zeros((2 * nb_q, n_shooting + 1))
    x_integrated_approx[:, 0] = x0
    x_integrated_real = np.zeros((2 * nb_q, n_shooting + 1))
    x_integrated_real[:, 0] = x0
    xdot_approx = np.zeros((2 * nb_q, n_shooting))
    xdot_real = np.zeros((2 * nb_q, n_shooting))
    M_real = np.zeros((n_shooting, nb_q, nb_q))
    N_real = np.zeros((n_shooting, nb_q))
    for i_node in range(n_shooting):
        if current_forward_dyn is None:
            x_integrated_approx = None
        elif (
                x_integrated_approx[0, i_node] > 0 and
                x_integrated_approx[0, i_node] < np.pi / 2 and
                x_integrated_approx[1, i_node] > 0 and
                x_integrated_approx[1, i_node] < 7/8 * np.pi and
                abs(x_integrated_approx[2, i_node]) < 10 * np.pi and
                abs(x_integrated_approx[3, i_node]) < 10 * np.pi
            ):
            x_integrated_approx[:, i_node + 1] = (
                RK4(
                    x_prev=x_integrated_approx[:, i_node],
                    u=u[:, i_node],
                    dt=dt,
                    motor_noise=np.zeros((nb_q,)),
                    forward_dyn_func=current_forward_dyn,
                    n_steps=5
                )
            )[-1, :]
        else:
            x_integrated_approx[:, i_node + 1] = np.nan
        x_integrated_real[:, i_node + 1] = (
            RK4(
                x_prev=x_integrated_real[:, i_node],
                u=u[:, i_node],
                dt=dt,
                motor_noise=np.zeros((nb_q,)),
                forward_dyn_func=real_forward_dyn,
                n_steps=5
            )
        )[-1, :]

        if current_forward_dyn is None:
            xdot_approx = None
        elif (
                x_integrated_real[0, i_node] > 0 and
                x_integrated_real[0, i_node] < np.pi / 2 and
                x_integrated_real[1, i_node] > 0 and
                x_integrated_real[1, i_node] < 7 / 8 * np.pi and
                abs(x_integrated_real[2, i_node]) < 10 * np.pi and
                abs(x_integrated_real[3, i_node]) < 10 * np.pi
        ):
            xdot_approx[:, i_node] = np.array(current_forward_dyn(
                x_integrated_real[:, i_node],
                u[:, i_node],
                np.zeros((nb_q,)),
            )).reshape(-1, )
        else:
            xdot_approx[:, i_node] = np.nan
        xdot_real[:, i_node] = np.array(real_forward_dyn(
            x_integrated_real[:, i_node],
            u[:, i_node],
            np.zeros((nb_q,)),
        )).reshape(-1, )
        M_real[i_node, :, :] = np.array(inv_mass_matrix_func(x_integrated_real[:, i_node]))
        N_real[i_node, :] = np.array(nl_effect_vector_func(x_integrated_real[:, i_node])).reshape(2, )
    return x_integrated_approx, x_integrated_real, xdot_approx, xdot_real, M_real, N_real


def integrate_MS(
        x_opt: np.ndarray,
        u: np.ndarray,
        dt: float,
        current_forward_dyn,
        real_forward_dyn: cas.Function,
):
    nb_q = 2
    n_steps = 5
    n_shooting = u.shape[1]
    x_integrated_approx_MS = np.zeros((2 * nb_q, (n_steps+1) * n_shooting))
    x_integrated_approx_MS[:, 0] = x_opt[:, 0]
    x_integrated_real_MS = np.zeros((2 * nb_q, (n_steps+1) * n_shooting))
    x_integrated_real_MS[:, 0] = x_opt[:, 0]
    for i_node in range(n_shooting):
        x_integrated_approx_MS[:, (n_steps+1) * i_node: (n_steps+1) * (i_node + 1)] = (
            RK4(
                x_prev=x_opt[:, i_node],
                u=u[:, i_node],
                dt=dt,
                motor_noise=np.zeros((nb_q,)),
                forward_dyn_func=current_forward_dyn,
                n_steps=n_steps
            )
        ).T
        x_integrated_real_MS[:, (n_steps+1) * i_node: (n_steps+1) * (i_node + 1)] = (
            RK4(
                x_prev=x_opt[:, i_node],
                u=u[:, i_node],
                dt=dt,
                motor_noise=np.zeros((nb_q,)),
                forward_dyn_func=real_forward_dyn,
                n_steps=n_steps
            )
        ).T

    return x_integrated_approx_MS, x_integrated_real_MS


def generate_random_data(nb_q, n_shooting, max_velocity: float = 10*np.pi, max_tau: float = 10):
    # Generate random data to compare against
    x0_this_time = np.array([
        np.random.uniform(0, np.pi / 2),
        np.random.uniform(0, 7 / 8 * np.pi),
        np.random.uniform(-max_velocity, max_velocity),
        np.random.uniform(-max_velocity, max_velocity),
    ])
    u_this_time = np.random.uniform(0, max_tau, (nb_q, n_shooting))
    # u_this_time = np.random.uniform(0, 0.1, (6, n_shooting))
    return x0_this_time, u_this_time


def sample_task_from_circle():
    """
    Get a random reaching task where the start and end targets are sampled from a circle centered at the home position.
    """
    circle_radius = 0.15  # 15 cm

    # Get the home position
    current_path = Path(__file__).parent
    model_path = f"{current_path}/../../StochasticOptimalControl/models/arm_model.bioMod"
    biorbd_model = biorbd.Biorbd(model_path)
    marker_func = biorbd_model.markers["end_effector"].forward_kinematics
    q_home = np.array([np.pi/4, np.pi/2])  # Wang et al. 2016
    home_position = get_dm_value(marker_func, [q_home])[:2, 0]

    # Get the random positions
    random_start_radius = np.random.uniform(0, circle_radius)
    random_start_angle = np.random.uniform(0, 2*np.pi)
    random_end_radius = np.random.uniform(0, circle_radius)
    random_end_angle = np.random.uniform(0, 2*np.pi)
    start_position = home_position + random_start_radius * np.array([np.cos(random_start_angle), np.sin(random_start_angle)])
    end_position = home_position + random_end_radius * np.array([np.cos(random_end_angle), np.sin(random_end_angle)])

    return start_position, end_position

def animate_reintegration(
        q_reintegrated: np.ndarray,
        muscles_opt: np.ndarray,
        ) -> None:

    import pyorerun
    n_shooting = q_reintegrated.shape[1] - 1
    final_time = 0.8

    # Add the model
    current_path = Path(__file__).parent
    model_path = f"{current_path}/../../StochasticOptimalControl/models/arm_model.bioMod"
    model = pyorerun.BiorbdModel(model_path)
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
    viz.add_animated_model(model, q_reintegrated, muscle_activations_intensity=pyoemg)

    # Play
    viz.rerun("Q reintegrated")