import pickle
import numpy as np
from datetime import datetime

from ..save_utils import get_print_tol, integrate_single_shooting_delay


def save_socp_delay(
    w_opt: np.ndarray,
    socp_delay: dict[str, any],
    save_path_socp_delay: str,
    tol: float,
    solver: any,
):

    n_q = socp_delay["model"].nb_q
    n_muscles = socp_delay["model"].nb_muscles
    n_references = socp_delay["model"].n_references
    n_random = socp_delay["model"].n_random

    # Parse ocp
    w0 = socp_delay["w0"]
    lbw = socp_delay["lbw"]
    ubw = socp_delay["ubw"]
    n_shooting = socp_delay["n_shooting"]
    final_time = socp_delay["final_time"]

    # Get optimization variables
    q_opt = np.zeros((n_q, n_random, n_shooting + 1))
    q0 = np.zeros((n_q, n_random, n_shooting + 1))
    lbq = np.zeros((n_q, n_random, n_shooting + 1))
    ubq = np.zeros((n_q, n_random, n_shooting + 1))

    qdot_opt = np.zeros((n_q, n_random, n_shooting + 1))
    qdot0 = np.zeros((n_q, n_random, n_shooting + 1))
    lbqdot = np.zeros((n_q, n_random, n_shooting + 1))
    ubqdot = np.zeros((n_q, n_random, n_shooting + 1))

    muscle_opt = np.zeros((n_muscles, n_shooting))
    muscle0 = np.zeros((n_muscles, n_shooting))
    lbmuscle = np.zeros((n_muscles, n_shooting))
    ubmuscle = np.zeros((n_muscles, n_shooting))

    k_fb_opt = np.zeros((n_q * n_references, n_shooting))
    k_fb0 = np.zeros((n_q * n_references, n_shooting))
    lbk_fb = np.zeros((n_q * n_references, n_shooting))
    ubk_fb = np.zeros((n_q * n_references, n_shooting))

    ref_fb_opt = np.zeros((n_references, n_shooting))
    ref_fb0 = np.zeros((n_references, n_shooting))
    lbref_fb = np.zeros((n_references, n_shooting))
    ubref_fb = np.zeros((n_references, n_shooting))

    x_opt = np.zeros((n_q * 2 * n_random, n_shooting + 1))
    u_opt = np.zeros((n_muscles + n_q * n_references + n_references, n_shooting))

    offset = 0
    for i_node in range(n_shooting + 1):
        for i_random in range(n_random):
            q_opt[:, i_random, i_node] = np.array(w_opt[offset : offset + n_q]).flatten()
            q0[:, i_random, i_node] = np.array(w0[offset : offset + n_q]).flatten()
            lbq[:, i_random, i_node] = np.array(lbw[offset : offset + n_q]).flatten()
            ubq[:, i_random, i_node] = np.array(ubw[offset : offset + n_q]).flatten()
            offset += n_q
        x_opt[: n_q * n_random, i_node] = q_opt[:, :, i_node].flatten(order="F")

        for i_random in range(n_random):
            qdot_opt[:, i_random, i_node] = np.array(w_opt[offset : offset + n_q]).flatten()
            qdot0[:, i_random, i_node] = np.array(w0[offset : offset + n_q]).flatten()
            lbqdot[:, i_random, i_node] = np.array(lbw[offset : offset + n_q]).flatten()
            ubqdot[:, i_random, i_node] = np.array(ubw[offset : offset + n_q]).flatten()
            offset += n_q
        x_opt[n_q * n_random :, i_node] = qdot_opt[:, :, i_node].flatten(order="F")

        if i_node < n_shooting:
            muscle_opt[:, i_node] = np.array(w_opt[offset : offset + n_muscles]).flatten()
            muscle0[:, i_node] = np.array(w0[offset : offset + n_muscles]).flatten()
            lbmuscle[:, i_node] = np.array(lbw[offset : offset + n_muscles]).flatten()
            ubmuscle[:, i_node] = np.array(ubw[offset : offset + n_muscles]).flatten()
            offset += n_muscles
            u_opt[:n_muscles, i_node] = muscle_opt[:, i_node].flatten()

            k_fb_opt[:, i_node] = np.array(w_opt[offset : offset + n_q * n_references]).flatten()
            k_fb0[:, i_node] = np.array(w0[offset : offset + n_q * n_references]).flatten()
            lbk_fb[:, i_node] = np.array(lbw[offset : offset + n_q * n_references]).flatten()
            ubk_fb[:, i_node] = np.array(ubw[offset : offset + n_q * n_references]).flatten()
            offset += n_q * n_references
            u_opt[n_muscles : n_muscles + n_q * n_references, i_node] = k_fb_opt[:, i_node].flatten()

            ref_fb_opt[:, i_node] = np.array(w_opt[offset : offset + n_references]).flatten()
            ref_fb0[:, i_node] = np.array(w0[offset : offset + n_references]).flatten()
            lbref_fb[:, i_node] = np.array(lbw[offset : offset + n_references]).flatten()
            ubref_fb[:, i_node] = np.array(ubw[offset : offset + n_references]).flatten()
            offset += n_references
            u_opt[n_muscles + n_q * n_references :, i_node] = ref_fb_opt[:, i_node].flatten()

    time_vector = np.linspace(0, final_time, n_shooting + 1)

    # Reintegrate the solution
    x_ee_delay_opt = np.zeros((n_q * 2 * n_random, n_shooting + 1))
    for i_node in range(n_shooting + 1):
        if i_node < socp_delay["model"].nb_frames_delay:
            # The delay is larger than the current time (not k anyways)
            x_ee_delay_opt[:, i_node] = x_opt[:, 0]
        else:
            x_ee_delay_opt[:, i_node] = x_opt[:, i_node - socp_delay["model"].nb_frames_delay]
    x_integrated = integrate_single_shooting_delay(socp_delay, x_opt, u_opt, x_ee_delay_opt)
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
    save_path_ocp = save_path_socp_delay.replace(".pkl", f"_{status}_{ocp_print_tol}.pkl")

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
