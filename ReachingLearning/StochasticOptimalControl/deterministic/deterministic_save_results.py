import pickle
import numpy as np
from datetime import datetime

from ..save_utils import integrate_single_shooting, get_print_tol


def save_ocp(
    w_opt: np.ndarray,
    ocp: dict[str, any],
    save_path_ocp: str,
    tol: float,
    solver: any,
):

    # Parse ocp
    w0 = ocp["w0"]
    lbw = ocp["lbw"]
    ubw = ocp["ubw"]
    n_shooting = ocp["n_shooting"]
    final_time = ocp["final_time"]

    # Get optimization variables
    q_opt = []
    q0 = []
    lbq = []
    ubq = []
    qdot_opt = []
    qdot0 = []
    lbqdot = []
    ubqdot = []
    muscle_opt = []
    muscle0 = []
    lbmuscle = []
    ubmuscle = []
    offset = 0
    for i_node in range(n_shooting + 1):

        q_opt += w_opt[offset : offset + 2].tolist()
        q0 += np.array(w0[offset : offset + 2]).flatten().tolist()
        lbq += np.array(lbw[offset : offset + 2]).flatten().tolist()
        ubq += np.array(ubw[offset : offset + 2]).flatten().tolist()
        offset += 2

        qdot_opt += w_opt[offset : offset + 2].tolist()
        qdot0 += np.array(w0[offset : offset + 2]).flatten().tolist()
        lbqdot += np.array(lbw[offset : offset + 2]).flatten().tolist()
        ubqdot += np.array(ubw[offset : offset + 2]).flatten().tolist()
        offset += 2

        if i_node < n_shooting:
            muscle_opt += w_opt[offset : offset + 6].tolist()
            muscle0 += np.array(w0[offset : offset + 6]).flatten().tolist()
            lbmuscle += np.array(lbw[offset : offset + 6]).flatten().tolist()
            ubmuscle += np.array(ubw[offset : offset + 6]).flatten().tolist()
            offset += 6

    q_opt = np.array(q_opt).reshape(2, n_shooting + 1, order="F")
    q0 = np.array(q0).reshape(2, n_shooting + 1, order="F")
    lbq = np.array(lbq).reshape(2, n_shooting + 1, order="F")
    ubq = np.array(ubq).reshape(2, n_shooting + 1, order="F")
    qdot_opt = np.array(qdot_opt).reshape(2, n_shooting + 1, order="F")
    qdot0 = np.array(qdot0).reshape(2, n_shooting + 1, order="F")
    lbqdot = np.array(lbqdot).reshape(2, n_shooting + 1, order="F")
    ubqdot = np.array(ubqdot).reshape(2, n_shooting + 1, order="F")
    muscle_opt = np.array(muscle_opt).reshape(6, n_shooting, order="F")
    muscle0 = np.array(muscle0).reshape(6, n_shooting, order="F")
    lbmuscle = np.array(lbmuscle).reshape(6, n_shooting, order="F")
    ubmuscle = np.array(ubmuscle).reshape(6, n_shooting, order="F")
    time_vector = np.linspace(0, final_time, n_shooting + 1)

    # Reintegrate the solution
    x_integrated = integrate_single_shooting(ocp, np.hstack((q_opt[:, 0], qdot_opt[:, 0])), muscle_opt)
    q_integrated = x_integrated[0:2, :]
    qdot_integrated = x_integrated[2:4, :]

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
    save_path_ocp = save_path_ocp.replace(".pkl", f"_{status}_{ocp_print_tol}.pkl")

    variable_data = {
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
