import pickle
import os

from ..utils import ExampleType, solve
from ..save_utils import get_print_tol
from .deterministic_OCP import prepare_ocp
from .deterministic_save_results import save_ocp
from .deterministic_plot import plot_ocp
from .deterministic_animate import animate_ocp
from .deterministic_confirm_solution import confirm_optimal_solution_ocp


def run_ocp(
    final_time: float,
    n_shooting: int,
    motor_noise_std: float,
    force_field_magnitude: float,
    example_type: ExampleType,
    n_threads: int,
    tol: float,
    n_simulations: int,
    RUN_OCP: bool,
    PLOT_FLAG: bool,
    ANIMATE_FLAG: bool,
):
    save_path_ocp = (
        f"../results/StochasticOptimalControl/ocp_forcefield{force_field_magnitude}_{example_type.value}.pkl"
    )

    ocp = prepare_ocp(
        final_time=final_time,
        n_shooting=n_shooting,
        example_type=example_type,
        force_field_magnitude=force_field_magnitude,
        n_threads=n_threads,
    )

    if RUN_OCP:
        print(
            "\nSolving OCP............................................................................................\n"
        )
        w_opt, f_opt, g_opt, solver = solve(ocp)
        variable_data = save_ocp(w_opt, ocp, save_path_ocp, tol, solver)
        confirm_optimal_solution_ocp(w_opt, f_opt, g_opt, ocp)
    else:
        ocp_print_tol = get_print_tol(tol)
        save_path_ocp = save_path_ocp.replace(".pkl", f"_CVG_{ocp_print_tol}.pkl")
        if not os.path.exists(save_path_ocp):
            raise FileNotFoundError(f"The file {save_path_ocp} does not exist, please run the optimization first.")

        with open(save_path_ocp, "rb") as file:
            variable_data = pickle.load(file)

    if PLOT_FLAG:
        plot_ocp(
            variable_data,
            ocp,
            motor_noise_std,
            force_field_magnitude,
            save_path_ocp,
            n_simulations=n_simulations,
        )

    if ANIMATE_FLAG:
        animate_ocp(
            final_time,
            n_shooting,
            variable_data["q_opt"],
            variable_data["muscle_opt"],
        )
