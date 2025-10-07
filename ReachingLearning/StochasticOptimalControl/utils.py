import numpy as np
import git
from datetime import date
import os
import pickle
import casadi as cas

from enum import Enum


class ExampleType(Enum):
    """
    Selection of the type of example to solve
    """

    CIRCLE = "CIRCLE"
    BAR = "BAR"


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

def solve(
        ocp: dict[str, any],
        max_iter: int=1000,
        tol: float=1e-6,
        hessian_approximation: str= "exact",  # or "limited-memory",
        output_file: str = None,
) -> np.ndarray:
    """ Solve the problem using IPOPT solver """

    # Extract the problem
    w = ocp["w"]
    j = ocp["j"]
    g = ocp["g"]
    w0 = ocp["w0"]
    lbw = ocp["lbw"]
    ubw = ocp["ubw"]
    lbg = ocp["lbg"]
    ubg = ocp["ubg"]

    # Set IPOPT options
    opts = {
        "ipopt.max_iter": max_iter,
        "ipopt.tol": tol,
        "ipopt.hessian_approximation": hessian_approximation,
        "ipopt.output_file": output_file,
    }

    # Create an NLP solver
    prob = {"f": j, "x": cas.vertcat(*w), "g": cas.vertcat(*g)}
    solver = cas.nlpsol("solver", "ipopt", prob, opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()

    # Check if the optimization converged
    if sol.stats()["success"] != True:
        print("WARNING: The solver did not converge")

    return w_opt