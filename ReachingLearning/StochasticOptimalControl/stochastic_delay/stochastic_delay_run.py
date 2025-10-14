import pickle
import os
import numpy as np

from ..utils import ExampleType, solve
from ..save_utils import get_print_tol
from .stochastic_delay_OCP import prepare_socp_delay
from .stochastic_delay_save_results import save_socp_delay
from .stochastic_delay_plot import plot_socp_delay
from .stochastic_delay_animate import animate_socp_delay


def run_socp_delay(
    final_time: float,
    n_shooting: int,
    motor_noise_std: float,
    wPq_std: float,
    wPqdot_std: float,
    force_field_magnitude: float,
    example_type: ExampleType,
    n_random: int,
    seed: int,
    n_threads: int,
    tol: float,
    n_simulations: int,
    RUN_SOCP_BASIC: bool,
    PLOT_FLAG: bool,
    ANIMATE_FLAG: bool,
):

    # TODO: see if we want to add noise on the initial state
    print_motor_noise_std = "{:1.1e}".format(motor_noise_std).replace(".", "p")
    print_wPq_std = "{:1.1e}".format(wPq_std).replace(".", "p")
    print_wPqdot_std = "{:1.1e}".format(wPqdot_std).replace(".", "p")
    save_path_socp_delay = f"../results/StochasticOptimalControl/socp_delay_forcefield{force_field_magnitude}_{example_type.value}_{print_motor_noise_std}_{print_wPq_std}_{print_wPqdot_std}.pkl"

    socp_delay = prepare_socp_delay(
        final_time=final_time,
        n_shooting=n_shooting,
        motor_noise_std=motor_noise_std,
        wPq_std=wPq_std,
        wPqdot_std=wPqdot_std,
        example_type=example_type,
        force_field_magnitude=force_field_magnitude,
        n_random=n_random,
        n_threads=n_threads,
        seed=seed,
    )

    if RUN_SOCP_BASIC:
        print(
            "\nSolving SOCP BASIC............................................................................................\n"
        )
        w_opt, solver = solve(socp_delay)
        variable_data = save_socp_delay(w_opt, socp_delay, save_path_socp_delay, tol, solver)
    else:
        ocp_print_tol = get_print_tol(tol)
        save_path_socp_delay = save_path_socp_delay.replace(".pkl", f"_CVG_{ocp_print_tol}.pkl")
        if not os.path.exists(save_path_socp_delay):
            raise FileNotFoundError(
                f"The file {save_path_socp_delay} does not exist, please run the optimization first."
            )

        with open(save_path_socp_delay, "rb") as file:
            variable_data = pickle.load(file)

    if PLOT_FLAG:
        plot_socp_delay(
            variable_data,
            socp_delay,
            motor_noise_std,
            force_field_magnitude,
            save_path_socp_delay,
            n_simulations=int(round(n_simulations / n_random)),
        )

    if ANIMATE_FLAG:
        animate_socp_delay(
            final_time,
            n_shooting,
            np.nanmean(variable_data["q_opt"], axis=1),
            variable_data["muscle_opt"],
        )
