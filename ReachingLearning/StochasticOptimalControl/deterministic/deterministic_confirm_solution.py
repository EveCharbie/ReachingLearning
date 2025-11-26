import numpy as np
import casadi as cas

from .deterministic_save_results import get_variables_from_vector, get_states_and_controls


def get_constraint_function(ocp: dict[str, any], constraint_name: str) -> cas.Function:
    """
    Get the indices of a specific constraint in the ocp dictionary.
    And create a casadi function for it.
    """
    indices = []
    for i, name in enumerate(ocp["g_names"]):
        if name == constraint_name:
            indices += [i]
    return cas.Function(constraint_name, [ocp["w"]], [cas.vertcat(ocp["g"][indices])])

def confirm_optimal_solution_ocp(
    w_opt: np.ndarray,
    f_opt: float,
    g_opt: np.ndarray,
    ocp: dict[str, any],
):
    """
    Confirm that the optimal solution has the same constraint and objective values as IPOPT.
    """

    n_q = ocp["model"].nb_q
    n_muscles = ocp["model"].nb_muscles
    n_shooting = ocp["n_shooting"]

    # Get optimization variables
    q_opt, qdot_opt, muscle_opt = get_variables_from_vector(
        n_q, n_shooting, n_muscles, w_opt
    )
    x_opt, u_opt = get_states_and_controls(
        n_q,
        n_shooting,
        n_muscles,
        q_opt,
        qdot_opt,
        muscle_opt,
    )

    # Check objective
    f_func = cas.Function("f_func", [ocp["w"]], [ocp["j"]])
    f_recomputed = f_func(w_opt)
    if np.abs(f_opt - f_recomputed) > 1e-8:
        raise RuntimeError("Recomputed objective does not match IPOPT objective to 1e-8.")

    # Check constraints
    g_func = cas.Function("g_func", [ocp["w"]], [ocp["g"]])
    g_recomputed = g_func(w_opt)
    if np.max(np.abs(g_opt - g_recomputed)) > 1e-8:
        raise RuntimeError("Recomputed constraints do not match IPOPT constraints to 1e-8.")
    if np.max(np.abs(g_recomputed)) > 1e-6:
        RuntimeError("Some constraints are larger than 1e-6. If there are no inequality constraint this is problematic.")

    # Check detailed constraints
    # Continuity
    for i_node in range(n_shooting):
        x_integrated = ocp["integration_func"](x_opt[:, i_node], u_opt[:, i_node])
        g_continuity = cas.reshape(x_integrated - x_opt[:, i_node+1], -1, 1)
        if np.max(np.abs(g_continuity)) > 1e-6:
            raise RuntimeError("Dynamics continuity constraints are not satisfied to 1e-6.")

    # Start on target
    g2_func = get_constraint_function(ocp, constraint_name="mean_start_on_target")
    g2_eval = g2_func(w_opt)
    if np.max(np.abs(g2_eval)) > 1e-8:
        raise RuntimeError("mean_start_on_target constraints are not satisfied to 1e-8.")

    # End on target
    g3_func = get_constraint_function(ocp, constraint_name="reach_target")
    g3_eval = g3_func(w_opt)
    if np.max(np.abs(g3_eval)) > 1e-8:
        raise RuntimeError("reach_target constraints are not satisfied to 1e-8.")