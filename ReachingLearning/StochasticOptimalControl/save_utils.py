import numpy as np
import os
import pickle
import git
from datetime import date


def get_git_version():

    # Save the version of bioptim and the date of the optimization for future reference
    repo = git.Repo(search_parent_directories=True)
    commit_id = str(repo.commit())
    branch = str(repo.active_branch)
    bioptim_version = repo.git.version_info
    git_date = repo.git.log("-1", "--format=%cd")
    version_dic = {
        "commit_id": commit_id,
        "git_date": git_date,
        "branch": branch,
        "bioptim_version": bioptim_version,
        "date_of_the_optimization": date.today().strftime("%b-%d-%Y-%H-%M-%S"),
    }
    return version_dic


def get_print_tol(tol: float):
    print_tol = "{:1.1e}".format(tol).replace(".", "p")
    return print_tol


def load_variable_data(save_path: str, tol: float):
    print_tol = get_print_tol(tol)
    save_full_path = save_path.replace(".pkl", f"_CVG_{print_tol}.pkl")
    if os.path.exists(save_full_path):
        with open(save_full_path, "rb") as file:
            variable_data = pickle.load(file)
            return variable_data
    else:
        raise FileNotFoundError(f"The file {save_full_path} does not exist, please run the optimization first.")


def integrate_single_shooting(ocp: dict[str, any], x_opt: np.ndarray, u_opt: np.ndarray):
    n_shooting = ocp["n_shooting"]

    x_integrated = np.zeros((x_opt.shape[0], n_shooting + 1))
    x_integrated[:, 0] = x_opt[:, 0]
    for i_node in range(n_shooting):
        x_integrated[:, i_node + 1] = (
            ocp["integration_func"](
                x=x_integrated[:, i_node],
                u=u_opt[:, i_node],
            )["x_next"]
            .full()
            .flatten()
        )
    return x_integrated
