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
    biorbd_model = biorbd.Model(model_path)
    nb_q = biorbd_model.nbQ()
    X = cas.MX.sym("x", nb_q * 2)
    U = cas.MX.sym("u", nb_q)
    motor_noise = cas.MX.sym("motor_noise", nb_q)

    xdot = cas.vertcat(X[nb_q:], biorbd_model.ForwardDynamics(X[:nb_q], X[nb_q:], U).to_mx())
    real_dynamics = cas.Function("forward_dynamics", [X, U, motor_noise], [xdot])

    inv_mass_matrix_func = cas.Function(
        "inv_mass_matrix",
        [X],
        [cas.inv(biorbd_model.massMatrix(X[:nb_q]).to_mx())],
    )
    nl_effect_vector_func = cas.Function(
        "nl_effect_vector",
        [X],
        [biorbd_model.NonLinearEffect(X[:nb_q], X[nb_q:]).to_mx()],
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
        else:
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
        else:
            xdot_approx[:, i_node] = np.array(current_forward_dyn(
                x_integrated_real[:, i_node],
                u[:, i_node],
                np.zeros((nb_q,)),
            )).reshape(-1, )
        xdot_real[:, i_node] = np.array(real_forward_dyn(
            x_integrated_real[:, i_node],
            u[:, i_node],
            np.zeros((nb_q,)),
        )).reshape(-1, )
        M_real[i_node, :, :] = np.array(inv_mass_matrix_func(x_integrated_real[:, i_node]))
        N_real[i_node, :] = np.array(nl_effect_vector_func(x_integrated_real[:, i_node])).reshape(2, )
    return x_integrated_approx, x_integrated_real, xdot_approx, xdot_real, M_real, N_real


def generate_random_data(nb_q, n_shooting):
    # Generate random data to compare against
    x0_this_time = np.array([
        np.random.uniform(0, np.pi / 2),
        np.random.uniform(0, 7 / 8 * np.pi),
        np.random.uniform(-5, 5),
        np.random.uniform(-5, 5),
    ])
    u_this_time = np.random.uniform(-1, 1, (nb_q, n_shooting))
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