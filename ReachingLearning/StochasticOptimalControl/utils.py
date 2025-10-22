import numpy as np
import casadi as cas
import biorbd_casadi as biorbd
from enum import Enum
import matplotlib.pyplot as plt

from .live_plot_utils import OnlineCallback


class ExampleType(Enum):
    """
    Selection of the type of example to solve
    """

    CIRCLE = "CIRCLE"
    BAR = "BAR"


def get_target_position(model) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the initial and final hand position from the bioMod.
    Do not use this in the OCP as it will unnecessarily slow down the optimization (only for DM values).
    """
    dymmy_variable = cas.MX.sym("dummy", 1)
    marker_start_index = biorbd.marker_index(model.biorbd_model, "hand_start")
    marker_end_index = biorbd.marker_index(model.biorbd_model, "hand_end")
    initial_pos = model.biorbd_model.marker(np.zeros((model.biorbd_model.nbQ(),)), marker_start_index).to_mx()[:2, 0]
    final_pos = model.biorbd_model.marker(np.zeros((model.biorbd_model.nbQ(),)), marker_end_index).to_mx()[:2, 0]
    func = cas.Function("hand_initial_position", [dymmy_variable], [initial_pos, final_pos])
    hand_initial_position, hand_final_position = func(0)
    return hand_initial_position, hand_final_position


def get_dm_value(function, values):
    """
    Get the DM value of a CasADi function.
    """
    variables = []
    for i_var in range(len(values)):
        variables += [cas.MX.sym(f"var_{i_var}", values[i_var].shape[0], 1)]
    func = cas.Function("temp_func", variables, [function(*variables)])
    output = func(*values)
    return output


def RK4(x_prev, u, dt, motor_noise, forward_dyn_func, n_steps=5):
    h = dt / n_steps
    x_all = cas.DM.zeros((n_steps + 1, x_prev.shape[0]))
    x_all[0, :] = x_prev
    for i_step in range(n_steps):
        k1 = forward_dyn_func(
            x_prev,
            u,
            motor_noise,
        )
        k2 = forward_dyn_func(
            x_prev + h / 2 * k1,
            u,
            motor_noise,
        )
        k3 = forward_dyn_func(
            x_prev + h / 2 * k2,
            u,
            motor_noise,
        )
        k4 = forward_dyn_func(
            x_prev + h * k3,
            u,
            motor_noise,
        )

        x_all[i_step + 1, :] = x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_prev = x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_all


def plot_jacobian(g: cas.MX, w: cas.MX):
    """Plot the Jacobian matrix using matplotlib"""
    sparsity = cas.jacobian_sparsity(g, w)
    plt.figure()
    plt.imshow(sparsity)
    plt.title("Jacobian Sparsity Pattern")
    plt.xlabel("Variables")
    plt.ylabel("Constraints")
    plt.savefig("jacobian_sparsity.png")
    plt.show()


def solve(
    ocp: dict[str, any],
    max_iter: int = 10000,
    tol: float = 1e-6,
    hessian_approximation: str = "exact",  # or "limited-memory",
    output_file: str = None,
    pre_optim_plot: bool = False,
) -> tuple[np.ndarray, dict[str, any]]:
    """Solve the problem using IPOPT solver"""

    # Extract the problem
    w = ocp["w"]
    j = ocp["j"]
    g = ocp["g"]
    w0 = ocp["w0"]
    lbw = ocp["lbw"]
    ubw = ocp["ubw"]
    lbg = ocp["lbg"]
    ubg = ocp["ubg"]
    g_names = ocp["g_names"]

    if len(g_names) != g.shape[0]:
        raise ValueError("The length of g_names must be equal to the number of constraints in g.")

    # Online callback for live plotting
    grad_f_func = cas.Function("grad_f", [w], [cas.gradient(j, w)])
    grad_g_func = cas.Function("grad_g", [w], [cas.jacobian(g, w).T])

    if pre_optim_plot:
        plot_jacobian(g, w)

    online_callback = OnlineCallback(
        nx=w.shape[0],
        ng=g.shape[0],
        grad_f_func=grad_f_func,
        grad_g_func=grad_g_func,
        g_names=g_names,
        ocp=ocp,
    )

    # Set IPOPT options
    opts = {
        "ipopt.max_iter": max_iter,
        "ipopt.tol": tol,
        "ipopt.linear_solver": "ma97",
        "ipopt.hessian_approximation": hessian_approximation,
        "ipopt.output_file": output_file,
        # "expand": True,
        "iteration_callback": online_callback,
    }

    # Create an NLP solver
    nlp = {"f": j, "x": w, "g": g}
    solver = cas.nlpsol("solver", "ipopt", nlp, opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()

    return w_opt, solver
