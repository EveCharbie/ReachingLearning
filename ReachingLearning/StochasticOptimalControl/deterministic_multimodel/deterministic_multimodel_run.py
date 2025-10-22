import pickle
import os
import numpy as np

from ..utils import ExampleType, solve
from ..save_utils import get_print_tol
from .deterministic_multimodel_OCP import prepare_ocp_multimodel
from .deterministic_multimodel_save_results import save_ocp_multimodel
from .deterministic_multimodel_plot import plot_ocp_multimodel
from .deterministic_multimodel_animate import animate_ocp_multimodel


def run_ocp_multimodel(
    final_time: float,
    n_shooting: int,
    motor_noise_std: float,
    example_type: ExampleType,
    n_random: int,
    seed: int,
    n_threads: int,
    tol: float,
    n_simulations: int,
    RUN_OCP_MULTIMODEL: bool,
    PLOT_FLAG: bool,
    ANIMATE_FLAG: bool,
):

    # TODO: see if we want to add noise on the initial state
    print_motor_noise_std = "{:1.1e}".format(motor_noise_std).replace(".", "p")
    save_path_ocp_multimodel = (
        f"../results/StochasticOptimalControl/ocp_multimodel_{example_type.value}_{print_motor_noise_std}.pkl"
    )

    ocp_multimodel = prepare_ocp_multimodel(
        final_time=final_time,
        n_shooting=n_shooting,
        motor_noise_std=motor_noise_std,
        example_type=example_type,
        n_random=n_random,
        n_threads=n_threads,
        seed=seed,
    )

    if RUN_OCP_MULTIMODEL:
        print(
            "\nSolving OCP MULTIMODEL............................................................................................\n"
        )
        w_opt, solver = solve(
            ocp_multimodel,
            tol=tol,
            output_file=save_path_ocp_multimodel.replace(".pkl", "_ipopt_output.txt"),
            pre_optim_plot=False,
        )
        variable_data = save_ocp_multimodel(w_opt, ocp_multimodel, save_path_ocp_multimodel, tol, solver)
    else:
        ocp_print_tol = get_print_tol(tol)
        save_path_ocp_multimodel = save_path_ocp_multimodel.replace(".pkl", f"_CVG_{ocp_print_tol}.pkl")
        if not os.path.exists(save_path_ocp_multimodel):
            raise FileNotFoundError(
                f"The file {save_path_ocp_multimodel} does not exist, please run the optimization first."
            )

        with open(save_path_ocp_multimodel, "rb") as file:
            variable_data = pickle.load(file)

    if PLOT_FLAG:
        plot_ocp_multimodel(
            variable_data,
            ocp_multimodel,
            save_path_ocp_multimodel,
            n_simulations=int(round(n_simulations / n_random)),
        )

    if ANIMATE_FLAG:
        animate_ocp_multimodel(
            final_time,
            n_shooting,
            np.nanmean(variable_data["q_opt"], axis=1),
        )
