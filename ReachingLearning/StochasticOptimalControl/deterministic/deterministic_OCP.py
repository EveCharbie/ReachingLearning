import casadi as cas
import numpy as np

from ..utils import ExampleType, inverse_kinematics_target
from ..constraints_utils import reach_target, start_on_target
from .deterministic_arm_model import DeterministicArmModel
from ...constants import TARGET_START_VAN_WOUWE, TARGET_END_VAN_WOUWE


def declare_variables(
    model,
    target_start: np.ndarray,
    target_end: np.ndarray,
    n_shooting: int,
    muscle_driven: bool = True,
) -> tuple[list[cas.MX], list[cas.MX], list[cas.MX], list[float], list[float], list[float]]:
    """
    Declare all variables (states and controls) and their initial guess
        - q: shoulder and elbow linear interpolation
        - qdot: shoulder and elbow 0
        - muscle activations: all 1e-6
    and bounds
        - q: shoulder in [0, np.pi/2], elbow in [0, 7/8 * np.pi]
        - qdot: shoulder and elbow in [-10*np.pi, 10*np.pi]
        - muscle activations: all in [1e-6, 1]
    """
    # Get the initial and final joint angles via inverse kinematics
    q_initial = inverse_kinematics_target(model.biorbd_model, target_start)  # 0.349065850398866, 2.245867726451909
    shoulder_pos_initial = q_initial[0]
    elbow_pos_initial = q_initial[1]
    q_final = inverse_kinematics_target(model.biorbd_model, target_end)  # 0.959931088596881, 1.159394851847144
    shoulder_pos_final = q_final[0]
    elbow_pos_final = q_final[1]

    joint_angles_init = np.zeros((2, n_shooting + 1))
    joint_angles_init[0, :] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting + 1)  # Shoulder
    joint_angles_init[1, :] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting + 1)  # Elbow

    x = []
    u = []
    w = []
    lbw = []
    ubw = []
    w0 = []
    for i_node in range(n_shooting + 1):
        q_i = cas.MX.sym(f"q_{i_node}", 2)
        qdot_i = cas.MX.sym(f"qdot_{i_node}", 2)
        x += [cas.vertcat(q_i, qdot_i)]
        w += [cas.vertcat(q_i, qdot_i)]
        if i_node == 0 or i_node == n_shooting:
            # No velocity at beginning and end of the movement
            lbw += [0, 0, 0, 0]
            ubw += [np.pi / 2, 7 / 8 * np.pi, 0, 0]
        else:
            lbw += [0, 0, -10 * np.pi, -10 * np.pi]
            ubw += [np.pi / 2, 7 / 8 * np.pi, 10 * np.pi, 10 * np.pi]
        w0 += [joint_angles_init[0, i_node], joint_angles_init[1, i_node], 0, 0]
        if i_node < n_shooting:
            if muscle_driven:
                muscle_i = cas.MX.sym(f"muscle_{i_node}", 6)
                u += [muscle_i]
                w += [muscle_i]
                lbw += [1e-6] * 6
                ubw += [1] * 6
                w0 += [1e-6] * 6
            else:
                tau_i = cas.MX.sym(f"tau_{i_node}", 2)
                u += [tau_i]
                w += [tau_i]
                lbw += [-100] * 2
                ubw += [100] * 2
                w0 += [1e-6] * 2
    return x, u, w, lbw, ubw, w0


def declare_dynamics_equation(model, x_single, u_single, dt):
    """
    Formulate discrete time dynamics
    Fixed step Runge-Kutta 4 integrator
    """

    n_steps = 5  # RK4 steps per interval
    h = dt / n_steps

    # Dynamics
    xdot = model.dynamics(x_single, u_single)
    dynamics_func = cas.Function(f"dynamics", [x_single, u_single], [xdot], ["x", "u"], ["xdot"])
    dynamics_func = dynamics_func.expand()

    # Integrator
    x_next = x_single[:]
    for j in range(n_steps):
        k1 = dynamics_func(x_next, u_single)
        k2 = dynamics_func(x_next + h / 2 * k1, u_single)
        k3 = dynamics_func(x_next + h / 2 * k2, u_single)
        k4 = dynamics_func(x_next + h * k3, u_single)
        x_next += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    integration_func = cas.Function("F", [x_single, u_single], [x_next], ["x", "u"], ["x_next"])
    integration_func = integration_func.expand()
    return dynamics_func, integration_func


def prepare_ocp(
    final_time: float,
    n_shooting: int,
    force_field_magnitude: float = 0,
    example_type=ExampleType.CIRCLE,
    n_threads: int = 12,
    forward_dynamics_func: cas.Function=None,
    target_start: np.ndarray = TARGET_START_VAN_WOUWE,
    target_end: np.ndarray = TARGET_END_VAN_WOUWE,
    muscle_driven: bool = True,
) -> dict[str, any]:

    # Model
    model = DeterministicArmModel(
        force_field_magnitude=force_field_magnitude,
        n_shooting=n_shooting,
        forward_dynamics_func=forward_dynamics_func,
        muscle_driven=muscle_driven,
    )

    # Variables
    x, u, w, lbw, ubw, w0 = declare_variables(model, target_start, target_end, n_shooting, muscle_driven=muscle_driven)

    # Start with an empty NLP
    j = 0
    g = []
    lbg = []
    ubg = []
    g_names = []

    # Dynamics
    dt = final_time / n_shooting
    dynamics_func, integration_func = declare_dynamics_equation(model, x_single=x[0], u_single=u[0], dt=dt)

    multi_threaded_integrator = integration_func.map(n_shooting, "thread", n_threads)

    # Initial constraint
    g_target, lbg_target, ubg_target = start_on_target(model, x[0], target_start)
    g += g_target
    lbg += lbg_target
    ubg += ubg_target
    g_names += [f"mean_start_on_target"] * len(lbg_target)

    # Multi-threaded continuity constraint
    x_integrated = multi_threaded_integrator(cas.horzcat(*x[:-1]), cas.horzcat(*u))
    g += [cas.reshape(x_integrated - cas.horzcat(*x[1:]), -1, 1)]
    lbg += [0] * (4 * n_shooting)
    ubg += [0] * (4 * n_shooting)
    g_names += [f"dynamics_continuity"] * (model.nb_q * 2 * n_shooting)

    # Objectives
    j += cas.sum2(cas.sum1(cas.horzcat(*u) ** 2 * dt / 2))  # Minimize muscle activations

    # Terminal constraint
    g_target, lbg_target, ubg_target = reach_target(model, x[-1], example_type, target_end)
    g += g_target
    lbg += lbg_target
    ubg += ubg_target
    g_names += [f"reach_target"] * len(lbg_target)

    ocp = {
        "model": model,
        "dynamics_func": dynamics_func,
        "integration_func": integration_func,
        "w": cas.vertcat(*w),
        "w0": cas.vertcat(*w0),
        "lbw": cas.vertcat(*lbw),
        "ubw": cas.vertcat(*ubw),
        "j": j,
        "g": cas.vertcat(*g),
        "lbg": cas.vertcat(*lbg),
        "ubg": cas.vertcat(*ubg),
        "g_names": g_names,
        "n_shooting": n_shooting,
        "final_time": final_time,
        "example_type": example_type,
        "force_field_magnitude": force_field_magnitude,
        "start_target": target_start,
        "end_target": target_end,
    }
    return ocp
