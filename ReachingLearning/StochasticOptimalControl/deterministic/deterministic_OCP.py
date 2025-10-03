import casadi as cas
import numpy as np

from ..utils import ExampleType
from .deterministic_arm_model import DeterministicArmModel


def declare_variables(n_shooting):
    x = []
    u = []
    for i_node in range(n_shooting + 1):
        q_i = cas.MX.sym(f"q_{i_node}", 2)
        qdot_i = cas.MX.sym(f"qdot_{i_node}", 2)
        x += [cas.vertcat(q_i, qdot_i)]
        if i_node < n_shooting:
            muscle_i = cas.MX.sym(f"muscle_{i_node}", 6)
            u += [muscle_i]
    return x, u


def declare_dynamics_equation(model, x, u, final_time, n_shooting):
    """
    Formulate discrete time dynamics
    Fixed step Runge-Kutta 4 integrator
    """

    n_steps = 5  # RK4 steps per interval

    # Variables
    x = x[0]
    u = u[0]

    # Dynamics
    xdot = model.dynamics(x, u)
    dynamics_func = cas.Function(f"dynamics", [x, u], [xdot], ["x", "u"], ["xdot"])

    # Integrator
    dt = final_time / n_shooting / n_steps
    x_next = x
    for j in range(n_steps):
        k1 = dynamics_func(x_next, u)
        k2 = dynamics_func(x_next + dt / 2 * k1, u)
        k3 = dynamics_func(x_next + dt / 2 * k2, u)
        k4 = dynamics_func(x_next + dt * k3, u)
        x_next += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    integration_func = cas.Function("F", [x, u], [x_next], ["x", "u"], ["x_next"])
    return dynamics_func, integration_func


def prepare_ocp(
    final_time: float,
    n_shooting: int,
    hand_final_position: np.ndarray,
    force_field_magnitude: float = 0,
    example_type=ExampleType.CIRCLE,
):

    # Model
    model = DeterministicArmModel(force_field_magnitude=force_field_magnitude)

    # Variables
    x, u = declare_variables(n_shooting)

    # Dynamics
    dynamics_func, integration_func = declare_dynamics_equation(model, x, u, final_time, n_shooting)

    # # Objectives / constraints
    # L = x1**2 + x2**2 + u**2

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []

    J = 0
    g = []
    lbg = []
    ubg = []

    # "Lift" initial conditions
    Xk = MX.sym("X0", 2)
    w += [Xk]
    lbw += [0, 1]
    ubw += [0, 1]
    w0 += [0, 1]

    # Formulate the NLP
    for k in range(n_shooting):
        # New NLP variable for the control
        Uk = MX.sym("U_" + str(k))
        w += [Uk]
        lbw += [-1]
        ubw += [1]
        w0 += [0]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk["xf"]
        J = J + Fk["qf"]

        # New NLP variable for state at end of interval
        Xk = MX.sym("X_" + str(k + 1), 2)
        w += [Xk]
        lbw += [-0.25, -inf]
        ubw += [inf, inf]
        w0 += [0, 0]

        # Add equality constraint
        g += [Xk_end - Xk]
        lbg += [0, 0]
        ubg += [0, 0]

    # Create an NLP solver
    prob = {"f": J, "x": cas.vertcat(*w), "g": cas.vertcat(*g)}
    solver = cas.nlpsol("solver", "ipopt", prob)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()

    # Plot the solution
    x1_opt = w_opt[0::3]
    x2_opt = w_opt[1::3]
    u_opt = w_opt[2::3]


# tgrid = [final_time/n_shooting*k for k in range(n_shooting+1)]
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.clf()
# plt.plot(tgrid, x1_opt, '--')
# plt.plot(tgrid, x2_opt, '-')
# plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
# plt.xlabel('t')
# plt.legend(['x1','x2','u'])
# plt.grid()
# plt.show()
