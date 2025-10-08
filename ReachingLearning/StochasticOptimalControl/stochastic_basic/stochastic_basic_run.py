import pickle
import os

from ..utils import ExampleType, solve
from ..save_utils import get_print_tol
from .stochastic_basic_OCP import prepare_socp_basic
from .stochastic_basic_save_results import save_socp_basic
from .stochastic_basic_plot import plot_socp_basic
from .stochastic_basic_animate import animate_socp_basic


def run_socp_basic(
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
    save_path_socp_basic = f"../results/StochasticOptimalControl/socp_basic_forcefield{force_field_magnitude}_{example_type.value}_{print_motor_noise_std}_{print_wPq_std}_{print_wPqdot_std}.pkl"

    socp_basic = prepare_socp_basic(
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
        w_opt, solver = solve(socp_basic)
        variable_data = save_socp_basic(w_opt, socp_basic, save_path_socp_basic, tol, solver)
    else:
        ocp_print_tol = get_print_tol(tol)
        save_path_socp_basic = save_path_socp_basic.replace(".pkl", f"_CVG_{ocp_print_tol}.pkl")
        if not os.path.exists(save_path_socp_basic):
            raise FileNotFoundError(
                f"The file {save_path_socp_basic} does not exist, please run the optimization first."
            )

        with open(save_path_socp_basic, "rb") as file:
            variable_data = pickle.load(file)

    if PLOT_FLAG:
        plot_socp_basic(
            variable_data,
            socp_basic,
            motor_noise_std,
            force_field_magnitude,
            save_path_socp_basic,
            n_simulations=int(round(n_simulations / n_random)),
        )

    # if ANIMATE_FLAG:
    #     animate_socp_basic(
    #         final_time,
    #         n_shooting,
    #         variable_data["q_opt"],
    #         variable_data["muscle_opt"],
    #     )
