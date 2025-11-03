import casadi as cas
import numpy as np

from ..utils import ExampleType, get_dm_value
from ..constraints_utils import (
    mean_start_on_target,
    mean_reach_target,
    ref_equals_mean_ref,
    mean_end_effector_velocity,
    residual_tau_equals_zero,
    TARGET_START,
    TARGET_END,
)
from ..objectives_utils import (
    reach_target_consistently,
    minimize_stochastic_efforts,
    minimize_gains,
    minimize_muscle_activations,
    minimize_residual_tau,
)
from .stochastic_basic_arm_model import StochasticBasicArmModel


def declare_variables(
    model: StochasticBasicArmModel,
) -> tuple[list[cas.MX], list[cas.MX], list[cas.MX], list[float], list[float], list[float]]:
    """
    Declare all variables (states and controls) and their initial guess
        - q: shoulder and elbow linear interpolation
        - qdot: shoulder and elbow 0
        - muscle activations: all 0.1
        - feedback gains: all 0.1
        - feedback_reference: hand position at initial guess
        - residual tau: all 0
    and bounds
        - q: shoulder in [0, np.pi/2], elbow in [0, 7/8 * np.pi]
        - qdot: shoulder and elbow in [-10*np.pi, 10*np.pi]
        - muscle activations: all in [1e-6, 1]
        - feedback gains: all in [-10, 10]
        - feedback_reference: all in [-1, 1]
        - residual tau: all in [-10, 10]
    """
    n_muscles = model.nb_muscles
    n_q = model.nb_q
    n_references = model.n_references# 2 hand position + 2 hand velocity
    n_random = model.n_random
    n_shooting = model.n_shooting

    # Optimized in Tom's version
    shoulder_pos_initial = 0.349065850398866
    elbow_pos_initial = 2.245867726451909
    shoulder_pos_final = 0.959931088596881
    elbow_pos_final = 1.159394851847144

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
        q_i = cas.MX.sym(f"q_{i_node}", n_q * n_random)
        qdot_i = cas.MX.sym(f"qdot_{i_node}", n_q * n_random)
        x += [cas.vertcat(q_i, qdot_i)]
        w += [cas.vertcat(q_i, qdot_i)]
        # q (range of motion)
        lbw += [0, 0] * n_random
        ubw += [np.pi / 2, 7 / 8 * np.pi] * n_random
        w0 += [joint_angles_init[0, i_node], joint_angles_init[1, i_node]] * n_random
        # qdot (no velocity at beginning of the movement)
        if i_node == 0:
            lbw += [0, 0] * n_random
            ubw += [0, 0] * n_random
        else:
            lbw += [-10 * np.pi, -10 * np.pi] * n_random
            ubw += [10 * np.pi, 10 * np.pi] * n_random
        w0 += [0, 0] * n_random
        if i_node < n_shooting:
            # Muscle activations
            muscle_i = cas.MX.sym(f"muscle_{i_node}", n_muscles)
            lbw += [1e-6] * n_muscles
            ubw += [1.0] * n_muscles
            w0 += [0.1] * n_muscles
            # Feedback gains
            k_fb_i = cas.MX.sym(f"k_fb_{i_node}", n_q * n_references)
            lbw += [-10] * (n_q * n_references)
            ubw += [10] * (n_q * n_references)
            w0 += [0.1] * (n_q * n_references)
            # Feedback reference
            ref_fb_i = cas.MX.sym(f"ref_fb_{i_node}", n_references)
            lbw += [-1] * n_references
            ubw += [1] * n_references
            ref_trajectory_init = get_dm_value(
                model.sensory_reference,
                [
                    joint_angles_init[:, i_node],  # q
                    np.zeros((n_q, )),  # qdot
                    np.zeros((n_references)),  # sensory_noise
                ],

            )
            w0 += [ref_trajectory_init]
            # Residual tau
            tau_i = cas.MX.sym(f"tau_{i_node}", n_q)
            lbw += [-10] * n_q
            ubw += [10] * n_q
            w0 += [0, 0]

            u += [cas.vertcat(muscle_i, k_fb_i, ref_fb_i, tau_i)]
            w += [cas.vertcat(muscle_i, k_fb_i, ref_fb_i, tau_i)]
    return x, u, w, lbw, ubw, w0


def declare_noises(n_shooting, n_random, motor_noise_magnitude, sensory_noise_magnitude):
    """
    Motor noise: 2 taus
    Sensory noise: 2 hand position + 2 hand velocity
    """
    n_q = 2
    n_references = 4

    noises_numerical = []
    for i_shooting in range(n_shooting):
        this_motor_noise_vector = np.zeros((n_q * n_random,))
        this_sensory_noise_vector = np.zeros((n_references * n_random,))
        for i_random in range(n_random):
            this_motor_noise_vector[n_q * i_random : n_q * (i_random + 1)] = np.random.normal(
                loc=np.zeros((n_q,)),
                scale=np.reshape(np.array(motor_noise_magnitude), (n_q,)),
                size=n_q,
            )
            this_sensory_noise_vector[n_references * i_random : n_references * (i_random + 1)] = np.random.normal(
                loc=np.zeros((n_references,)),
                scale=np.reshape(np.array(sensory_noise_magnitude), (n_references,)),
                size=n_references,
            )
        noises_numerical += [cas.vertcat(this_motor_noise_vector, this_sensory_noise_vector)]
    noises_single = cas.MX.sym("noises_single", (n_q + n_references) * n_random)
    return noises_numerical, noises_single


def declare_dynamics_equation(model, x_single, u_single, noises_single, dt):
    """
    Formulate discrete time dynamics
    Fixed step Runge-Kutta 4 integrator
    """

    n_steps = 5  # RK4 steps per interval
    h = dt / n_steps

    # Dynamics
    xdot = model.dynamics(x_single, u_single, noises_single)
    dynamics_func = cas.Function(
        f"dynamics", [x_single, u_single, noises_single], [xdot], ["x", "u", "noise"], ["xdot"]
    )
    dynamics_func = dynamics_func.expand()

    # Integrator
    x_next = x_single[:]
    for j in range(n_steps):
        k1 = dynamics_func(x_next, u_single, noises_single)
        k2 = dynamics_func(x_next + h / 2 * k1, u_single, noises_single)
        k3 = dynamics_func(x_next + h / 2 * k2, u_single, noises_single)
        k4 = dynamics_func(x_next + h * k3, u_single, noises_single)
        x_next += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    integration_func = cas.Function("F", [x_single, u_single, noises_single], [x_next], ["x", "u", "noise"], ["x_next"])
    integration_func = integration_func.expand()
    return dynamics_func, integration_func


def prepare_socp_basic(
    final_time: float,
    n_shooting: int,
    motor_noise_std: float,
    wPq_std: float,
    wPqdot_std: float,
    force_field_magnitude: float = 0,
    example_type=ExampleType.CIRCLE,
    n_random: int = 20,
    n_threads: int = 12,
    seed: int = 0,
) -> dict[str, any]:

    # Fix the random seed for the noise generation
    np.random.seed(seed)

    dt = final_time / n_shooting

    # Model
    model = StochasticBasicArmModel(
        motor_noise_std=motor_noise_std,
        wPq_std=wPq_std,
        wPqdot_std=wPqdot_std,
        force_field_magnitude=force_field_magnitude,
        n_random=n_random,
        n_shooting=n_shooting,
    )

    # Variables
    x, u, w, lbw, ubw, w0 = declare_variables(model)
    noises_numerical, noises_single = declare_noises(
        n_shooting, n_random, model.motor_noise_magnitude, model.hand_sensory_noise_magnitude
    )

    # Start with an empty NLP
    j = 0
    g = []
    lbg = []
    ubg = []
    g_names = []

    # Dynamics
    dynamics_func, integration_func = declare_dynamics_equation(
        model, x_single=x[0], u_single=u[0], noises_single=noises_single, dt=dt
    )

    multi_threaded_integrator = integration_func.map(n_shooting, "thread", n_threads)

    # Initial constraint
    g_target, lbg_target, ubg_target = mean_start_on_target(model, x[0])
    g += g_target
    lbg += lbg_target
    ubg += ubg_target
    g_names += [f"mean_start_on_target"] * len(lbg_target)

    # Multi-threaded continuity constraint
    x_integrated = multi_threaded_integrator(cas.horzcat(*x[:-1]), cas.horzcat(*u), cas.horzcat(*noises_numerical))
    g_continuity = cas.reshape(x_integrated - cas.horzcat(*x[1:]), -1, 1)
    g += [g_continuity]
    lbg += [0] * ((model.nb_q * 2 * n_random) * n_shooting)
    ubg += [0] * ((model.nb_q * 2 * n_random) * n_shooting)
    g_names += [f"dynamics_continuity"] * ((model.nb_q * 2 * n_random) * n_shooting)

    # Objectives
    for i_node in range(n_shooting):
        j += minimize_stochastic_efforts(model, x[i_node], u[i_node], noises_numerical[i_node]) * dt / 2
        j += minimize_muscle_activations(model, u[i_node]) * dt / 2
        j += minimize_residual_tau(model, u[i_node]) * 10 * dt / 2
        j += minimize_gains(model, u[i_node]) * dt / 10  # Regularization
        j += cas.sum1(ref_equals_mean_ref(model, x[i_node], u[i_node])[0]**2) * 10 * dt / 2
    j += reach_target_consistently(model, x[-1], example_type)

    # # Constraints
    # for i_node in range(n_shooting):
    #     # Reference equality constraint
    #     reference_constraint = ref_equals_mean_ref(model, x[i_node], u[i_node])
    #     g += reference_constraint
    #     lbg += [0] * model.n_references
    #     ubg += [0] * model.n_references
    #     g_names += [f"ref_equals_mean_ref"] * model.n_references
    #
    # #     # # Null torque constraint
    # #     # g += residual_tau_equals_zero(model, u[i_node])
    # #     # lbg += [0] * model.nb_q
    # #     # ubg += [0] * model.nb_q
    # #     # g_names += [f"residual_tau_equals_zero"] * model.nb_q

    # Terminal constraint
    g_target, lbg_target, ubg_target = mean_reach_target(model, x[-1], example_type)
    g += g_target
    lbg += lbg_target
    ubg += ubg_target
    g_names += [f"mean_reach_target"] * len(lbg_target)
    g_target = mean_end_effector_velocity(model, x[-1])
    g += g_target
    lbg += [0, 0]
    ubg += [0, 0]
    g_names += [f"mean_end_effector_velocity"] * 2

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
    }
    return ocp
