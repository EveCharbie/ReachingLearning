import pickle
import numpy as np
from datetime import datetime

from ..save_utils import get_print_tol, integrate_single_shooting


def get_variables_from_vector(n_q, n_random, n_shooting, n_muscles, n_references, vector):

    # Get optimization variables
    q_opt = np.zeros((n_q, n_random, n_shooting + 1))
    qdot_opt = np.zeros((n_q, n_random, n_shooting + 1))
    muscle_opt = np.zeros((n_muscles, n_shooting))
    k_fb_opt = np.zeros((n_q * n_references, n_shooting))
    ref_fb_opt = np.zeros((n_references, n_shooting))
    tau_opt = np.zeros((n_q, n_shooting))

    offset = 0
    for i_node in range(n_shooting + 1):
        for i_random in range(n_random):
            q_opt[:, i_random, i_node] = np.array(vector[offset : offset + n_q]).flatten()
            offset += n_q

        for i_random in range(n_random):
            qdot_opt[:, i_random, i_node] = np.array(vector[offset : offset + n_q]).flatten()
            offset += n_q

        if i_node < n_shooting:
            muscle_opt[:, i_node] = np.array(vector[offset : offset + n_muscles]).flatten()
            offset += n_muscles

            k_fb_opt[:, i_node] = np.array(vector[offset : offset + n_q * n_references]).flatten()
            offset += n_q * n_references

            ref_fb_opt[:, i_node] = np.array(vector[offset : offset + n_references]).flatten()
            offset += n_references

            tau_opt[:, i_node] = np.array(vector[offset : offset + n_q]).flatten()
            offset += n_q

    return q_opt, qdot_opt, muscle_opt, k_fb_opt, ref_fb_opt, tau_opt

def get_states_and_controls(
        n_q,
        n_random,
        n_shooting,
        n_muscles,
        n_references,
        q_opt,
        qdot_opt,
        muscle_opt,
        k_fb_opt,
        ref_fb_opt,
        tau_opt,
):
    # Get optimization variables
    x_opt = np.zeros((n_q * 2 * n_random, n_shooting + 1))
    u_opt = np.zeros((n_muscles + n_q * n_references + n_references + n_q, n_shooting))

    for i_node in range(n_shooting + 1):
        x_opt[: n_q * n_random, i_node] = q_opt[:, :, i_node].flatten(order="F")
        x_opt[n_q * n_random:, i_node] = qdot_opt[:, :, i_node].flatten(order="F")

        if i_node < n_shooting:
            u_opt[:n_muscles, i_node] = muscle_opt[:, i_node].flatten()
            muscle_offset = n_muscles
            u_opt[muscle_offset: muscle_offset + n_q * n_references, i_node] = k_fb_opt[:, i_node].flatten()
            gain_offset = muscle_offset + n_q * n_references
            u_opt[gain_offset: gain_offset + n_references, i_node] = ref_fb_opt[:, i_node].flatten()
            reference_offset = gain_offset + n_references
            u_opt[reference_offset: reference_offset + n_q, i_node] = tau_opt[:, i_node].flatten()
            tau_offset = reference_offset + n_q

    return x_opt, u_opt

def save_socp_basic(
    w_opt: np.ndarray,
    socp_basic: dict[str, any],
    save_path_socp_basic: str,
    tol: float,
    solver: any,
):

    n_q = socp_basic["model"].nb_q
    n_muscles = socp_basic["model"].nb_muscles
    n_references = socp_basic["model"].n_references
    n_random = socp_basic["model"].n_random

    # Parse ocp
    w0 = socp_basic["w0"]
    lbw = socp_basic["lbw"]
    ubw = socp_basic["ubw"]
    n_shooting = socp_basic["n_shooting"]
    final_time = socp_basic["final_time"]

    # Get optimization variables
    q_opt, qdot_opt, muscle_opt, k_fb_opt, ref_fb_opt, tau_opt = get_variables_from_vector(n_q, n_random, n_shooting, n_muscles, n_references, w_opt)
    q0, qdot0, muscle0, k_fb0, ref_fb0, tau0 = get_variables_from_vector(n_q, n_random, n_shooting, n_muscles, n_references, w0)
    lbq, lbqdot, lbmuscle, lbk_fb, lbref_fb, lbtau = get_variables_from_vector(n_q, n_random, n_shooting, n_muscles, n_references, lbw)
    ubq, ubqdot, ubmuscle, ubk_fb, ubref_fb, ubtau = get_variables_from_vector(n_q, n_random, n_shooting, n_muscles, n_references, ubw)
    x_opt, u_opt = get_states_and_controls(
        n_q,
        n_random,
        n_shooting,
        n_muscles,
        n_references,
        q_opt,
        qdot_opt,
        muscle_opt,
        k_fb_opt,
        ref_fb_opt,
        tau_opt,
    )

    time_vector = np.linspace(0, final_time, n_shooting + 1)

    # Reintegrate the solution
    x_integrated = integrate_single_shooting(socp_basic, x_opt, u_opt)
    q_integrated = np.zeros((n_q, n_random, n_shooting + 1))
    qdot_integrated = np.zeros((n_q, n_random, n_shooting + 1))
    for i_node in range(n_shooting + 1):
        for i_random in range(n_random):
            q_integrated[:, i_random, i_node] = x_integrated[i_random * n_q : (i_random + 1) * n_q, i_node]
            qdot_integrated[:, i_random, i_node] = x_integrated[
                n_q * n_random + i_random * n_q : n_q * n_random + (i_random + 1) * n_q, i_node
            ]

    # Other info oin the optimization process
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    computational_time = solver.stats()["t_proc_total"]
    nb_iterations = solver.stats()["iter_count"]

    # Check if the optimization converged
    if solver.stats()["success"]:
        status = "CVG"
    else:
        status = "DVG"
    ocp_print_tol = get_print_tol(tol)
    save_path_ocp = save_path_socp_basic.replace(".pkl", f"_{status}_{ocp_print_tol}.pkl")

    variable_data = {
        "x_opt": x_opt,
        "u_opt": u_opt,
        "q_opt": q_opt,
        "q0": q0,
        "lbq": lbq,
        "ubq": ubq,
        "q_integrated": q_integrated,
        "qdot_opt": qdot_opt,
        "qdot0": qdot0,
        "lbqdot": lbqdot,
        "ubqdot": ubqdot,
        "qdot_integrated": qdot_integrated,
        "muscle_opt": muscle_opt,
        "muscle0": muscle0,
        "lbmuscle": lbmuscle,
        "ubmuscle": ubmuscle,
        "k_fb_opt": k_fb_opt,
        "k_fb0": k_fb0,
        "lbk_fb": lbk_fb,
        "ubk_fb": ubk_fb,
        "ref_fb_opt": ref_fb_opt,
        "ref_fb0": ref_fb0,
        "lbref_fb": lbref_fb,
        "ubref_fb": ubref_fb,
        "tau_opt": tau_opt,
        "tau0": tau0,
        "lbtau": lbtau,
        "ubtau": ubtau,
        "time_vector": time_vector,
        "computational_time": computational_time,
        "nb_iterations": nb_iterations,
        "current_time": current_time,
    }

    # --- Save --- #
    with open(save_path_ocp, "wb") as file:
        pickle.dump(variable_data, file)

    print("Saved : ", save_path_ocp)

    return variable_data
