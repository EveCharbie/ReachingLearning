import matplotlib.pyplot as plt
import numpy as np

from ..utils import get_target_position, get_dm_value
from ..plot_utils import set_columns_suptitles


SOCP_DELAY_color = "#F18F01"
# "#A469F1"
# "#06b0f0"


def hand_position(socp_delay, q):
    hand_pos = get_dm_value(socp_delay["model"].end_effector_position, [q])
    return np.reshape(hand_pos[:2], (2,))


def hand_velocity(socp_delay, q, qdot):
    hand_velo = get_dm_value(socp_delay["model"].end_effector_velocity, [q, qdot])
    return np.reshape(hand_velo[:2], (2,))


def plot_states(variable_data, socp_delay, save_path_socp_delay):

    n_shooting = socp_delay["n_shooting"]
    time_vector = variable_data["time_vector"]

    # Set up the plots
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    axs[0, 0].set_title("Shoulder angle")
    axs[0, 0].set_ylabel("Position [rad]")
    axs[0, 1].set_title("Elbow angle")
    axs[1, 0].set_ylabel("Velocity [rad/s]")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 1].set_xlabel("Time [s]")
    axs[0, 0].set_ylim([np.min(variable_data["lbq"]) - 0.1, np.max(variable_data["ubq"]) + 0.1])
    axs[0, 1].set_ylim([np.min(variable_data["lbq"]) - 0.1, np.max(variable_data["ubq"]) + 0.1])
    axs[1, 0].set_ylim([np.min(variable_data["lbqdot"]) - 1, np.max(variable_data["ubqdot"]) + 1])
    axs[1, 1].set_ylim([np.min(variable_data["lbqdot"]) - 1, np.max(variable_data["ubqdot"]) + 1])
    axs[0, 0].set_xlim([0, time_vector[-1]])
    axs[0, 1].set_xlim([0, time_vector[-1]])
    axs[1, 0].set_xlim([0, time_vector[-1]])
    axs[1, 1].set_xlim([0, time_vector[-1]])

    # Optimization variables
    for i_random in range(socp_delay["model"].n_random):
        axs[0, 0].plot(time_vector, variable_data["q_opt"][0, i_random, :], ".", markersize=1, color=SOCP_DELAY_color)
        axs[0, 1].plot(time_vector, variable_data["q_opt"][1, i_random, :], ".", markersize=1, color=SOCP_DELAY_color)
        axs[1, 0].plot(
            time_vector, variable_data["qdot_opt"][0, i_random, :], ".", markersize=1, color=SOCP_DELAY_color
        )
        axs[1, 1].plot(
            time_vector, variable_data["qdot_opt"][1, i_random, :], ".", markersize=1, color=SOCP_DELAY_color
        )

    # Reintegration
    for i_random in range(socp_delay["model"].n_random):
        axs[0, 0].plot(
            time_vector, variable_data["q_integrated"][0, i_random, :], "-", linewidth=0.5, color=SOCP_DELAY_color
        )
        axs[0, 1].plot(
            time_vector, variable_data["q_integrated"][1, i_random, :], "-", linewidth=0.5, color=SOCP_DELAY_color
        )
        axs[1, 0].plot(
            time_vector, variable_data["qdot_integrated"][0, i_random, :], "-", linewidth=0.5, color=SOCP_DELAY_color
        )
        axs[1, 1].plot(
            time_vector, variable_data["qdot_integrated"][1, i_random, :], "-", linewidth=0.5, color=SOCP_DELAY_color
        )

    # Bounds
    axs[0, 0].fill_between(
        time_vector, np.ones((n_shooting + 1,)) * -10, variable_data["lbq"][0, 0, :], color="lightgrey"
    )
    axs[0, 0].fill_between(
        time_vector, variable_data["ubq"][0, 0, :], np.ones((n_shooting + 1,)) * 10, color="lightgrey"
    )
    axs[0, 1].fill_between(
        time_vector, np.ones((n_shooting + 1,)) * -10, variable_data["lbq"][1, 0, :], color="lightgrey"
    )
    axs[0, 1].fill_between(
        time_vector, variable_data["ubq"][1, 0, :], np.ones((n_shooting + 1,)) * 10, color="lightgrey"
    )
    axs[1, 0].fill_between(
        time_vector,
        np.ones((n_shooting + 1,)) * -100,
        variable_data["lbqdot"][0, 0, :],
        color="lightgrey",
    )
    axs[1, 0].fill_between(
        time_vector, variable_data["ubqdot"][0, 0, :], np.ones((n_shooting + 1,)) * 100, color="lightgrey"
    )
    axs[1, 1].fill_between(
        time_vector,
        np.ones((n_shooting + 1,)) * -100,
        variable_data["lbqdot"][1, 0, :],
        color="lightgrey",
    )
    axs[1, 1].fill_between(
        time_vector, variable_data["ubqdot"][1, 0, :], np.ones((n_shooting + 1,)) * 100, color="lightgrey"
    )

    save_path_fig = save_path_socp_delay.replace(".pkl", "_plot_states.png").replace("/results/", "/figures/")
    plt.savefig(save_path_fig)
    plt.show()
    # plt.close()


def plot_controls(variable_data, socp_delay, save_path_socp_delay):

    n_shooting = socp_delay["n_shooting"]
    time_vector = variable_data["time_vector"][:-1]

    # Set up the plots
    fig, axs = plt.subplots(2, 3, figsize=(6, 6))
    fig, axs = set_columns_suptitles(fig, axs)
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 2].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Flexor Activation")
    axs[1, 0].set_ylabel("Extensor Activation")
    axs[0, 0].set_ylim([np.min(variable_data["lbmuscle"]) - 0.1, np.max(variable_data["ubmuscle"]) + 0.1])
    axs[0, 1].set_ylim([np.min(variable_data["lbmuscle"]) - 0.1, np.max(variable_data["ubmuscle"]) + 0.1])
    axs[0, 2].set_ylim([np.min(variable_data["lbmuscle"]) - 0.1, np.max(variable_data["ubmuscle"]) + 0.1])
    axs[1, 0].set_ylim([np.min(variable_data["lbmuscle"]) - 0.1, np.max(variable_data["ubmuscle"]) + 0.1])
    axs[1, 1].set_ylim([np.min(variable_data["lbmuscle"]) - 0.1, np.max(variable_data["ubmuscle"]) + 0.1])
    axs[1, 2].set_ylim([np.min(variable_data["lbmuscle"]) - 0.1, np.max(variable_data["ubmuscle"]) + 0.1])
    axs[0, 0].set_xlim([0, time_vector[-1]])
    axs[0, 1].set_xlim([0, time_vector[-1]])
    axs[0, 2].set_xlim([0, time_vector[-1]])
    axs[1, 0].set_xlim([0, time_vector[-1]])
    axs[1, 1].set_xlim([0, time_vector[-1]])
    axs[1, 2].set_xlim([0, time_vector[-1]])

    # Optimization variables
    axs[0, 0].plot(time_vector, variable_data["muscle_opt"][2, :], ".", markersize=1, color=SOCP_DELAY_color)
    axs[0, 0].set_title("Deltoid Anterior")
    axs[0, 1].plot(time_vector, variable_data["muscle_opt"][0, :], ".", markersize=1, color=SOCP_DELAY_color)
    axs[0, 1].set_title("Brachialis")
    axs[0, 2].plot(time_vector, variable_data["muscle_opt"][4, :], ".", markersize=1, color=SOCP_DELAY_color)
    axs[0, 2].set_title("Biceps")
    axs[1, 0].plot(time_vector, variable_data["muscle_opt"][3, :], ".", markersize=1, color=SOCP_DELAY_color)
    axs[1, 0].set_title("Deltoid Posterior")
    axs[1, 1].plot(time_vector, variable_data["muscle_opt"][1, :], ".", markersize=1, color=SOCP_DELAY_color)
    axs[1, 1].set_title("Triceps Lateral")
    axs[1, 2].plot(time_vector, variable_data["muscle_opt"][5, :], ".", markersize=1, color=SOCP_DELAY_color)
    axs[1, 2].set_title("Triceps Long")

    # Bounds
    if np.any(np.abs(variable_data["lbmuscle"] - 1e-6) > 1e-6):
        raise RuntimeError("Muscle lower bound is not 1e-6, please update the plotting code")
    if np.any(np.abs(variable_data["ubmuscle"] - 1) > 1e-6):
        raise RuntimeError("Muscle upper bound is not 1, please update the plotting code")

    for i_ax in range(2):
        for j_ax in range(3):
            axs[i_ax, j_ax].fill_between(
                time_vector,
                np.ones((n_shooting,)) * -0.1,
                variable_data["lbmuscle"][i_ax * 2 + j_ax, :],
                color="lightgrey",
            )
            axs[i_ax, j_ax].fill_between(
                time_vector,
                variable_data["ubmuscle"][i_ax * 2 + j_ax, :],
                np.ones((n_shooting,)) * 1.1,
                color="lightgrey",
            )

    plt.tight_layout()
    save_path_fig = save_path_socp_delay.replace(".pkl", "_plot_controls.png").replace("/results/", "/figures/")
    plt.savefig(save_path_fig)
    plt.show()
    # plt.close()

    # TODO: plot k
    print(f"Mean k_fb : {np.mean(variable_data['k_fb_opt'])}")
    print(f"Min k_fb : {np.min(variable_data['k_fb_opt'])} ({np.argmin(np.min(variable_data['k_fb_opt']))})")
    print(f"Max k_fb : {np.max(variable_data['k_fb_opt'])} ({np.argmax(np.min(variable_data['k_fb_opt']))})")


def plot_hand_trajectories(variable_data, socp_delay, n_simulations, save_path_socp_delay):

    n_shooting = socp_delay["n_shooting"]
    final_time = socp_delay["final_time"]
    n_random = socp_delay["model"].n_random
    n_muscles = socp_delay["model"].nb_muscles
    q_offset = socp_delay["model"].q_offset
    n_q = socp_delay["model"].nb_q
    n_noises = (socp_delay["model"].n_references + socp_delay["model"].nb_muscles) * n_random
    noise_magnitude = np.hstack(
        (
            np.array(
                np.array(socp_delay["model"].motor_noise_magnitude)
                .reshape(
                    -1,
                )
                .tolist()
                * n_random
            ),
            np.array(
                np.array(socp_delay["model"].hand_sensory_noise_magnitude)
                .reshape(
                    -1,
                )
                .tolist()
                * n_random
            ),
        )
    )
    nb_frames_delay = socp_delay["model"].nb_frames_delay

    # Reintegrate the solution with noise
    x_simulated = np.zeros((n_simulations * n_random, 2 * n_q, n_shooting + 1))
    hand_pos_simulated = np.zeros((n_simulations * n_random, 2, n_shooting + 1))
    hand_vel_simulated = np.zeros((n_simulations * n_random, 2, n_shooting + 1))
    for i_simulation in range(n_simulations):
        print(f"Running socp_delay noised simulation {i_simulation}")
        np.random.seed(i_simulation)
        for i_random in range(n_random):
            x_simulated[i_simulation * n_random + i_random, :n_q, 0] = variable_data["x_opt"][
                i_random * n_q : (i_random + 1) * n_q, 0
            ]
            x_simulated[i_simulation * n_random + i_random, n_q:, 0] = variable_data["x_opt"][
                n_q * n_random + i_random * n_q : n_q * n_random + (i_random + 1) * n_q, 0
            ]
        for i_node in range(n_shooting):
            x_prev = np.zeros((n_q * 2 * n_random))
            for i_random in range(n_random):
                x_prev[i_random * n_q : (i_random + 1) * n_q] = x_simulated[
                    i_simulation * n_random + i_random, :n_q, i_node
                ]
                x_prev[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q] = x_simulated[
                    i_simulation * n_random + i_random,
                    n_q:,
                    i_node,
                ]
            u_this_time = variable_data["u_opt"][:, i_node]
            noise_this_time = np.random.normal(0, noise_magnitude, n_noises)
            if i_node < nb_frames_delay:
                x_ee_delay = np.zeros((n_q * 2 * n_random))
            else:
                x_ee_delay = x_simulated[
                    i_simulation * n_random : (i_simulation + 1) * n_random, :, i_node - nb_frames_delay
                ]
            x_next = socp_delay["integration_func"](x_prev, u_this_time, x_ee_delay, noise_this_time)
            for i_random in range(n_random):
                x_simulated[i_simulation * n_random + i_random, :n_q, i_node + 1] = np.reshape(
                    x_next[i_random * n_q : (i_random + 1) * n_q, 0], (-1,)
                )
                x_simulated[i_simulation * n_random + i_random, n_q:, i_node + 1] = np.reshape(
                    x_next[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q, 0], (-1,)
                )

                hand_pos_simulated[i_simulation * n_random + i_random, :, i_node] = hand_position(
                    socp_delay, x_prev[i_random * n_q : (i_random + 1) * n_q]
                )
                hand_vel_simulated[i_simulation * n_random + i_random, :, i_node] = hand_velocity(
                    socp_delay,
                    x_prev[i_random * n_q : (i_random + 1) * n_q],
                    x_prev[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q],
                )

        # Final point
        x_prev = np.zeros((n_q * 2 * n_random))
        for i_random in range(n_random):
            x_prev[i_random * n_q : (i_random + 1) * n_q] = x_simulated[
                i_simulation * n_random + i_random, :n_q, i_node + 1
            ]
            x_prev[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q] = x_simulated[
                i_simulation * n_random + i_random,
                n_q:,
                i_node + 1,
            ]
        for i_random in range(n_random):
            hand_pos_simulated[i_simulation * n_random + i_random, :, i_node + 1] = hand_position(
                socp_delay, x_prev[i_random * n_q : (i_random + 1) * n_q]
            )
            hand_vel_simulated[i_simulation * n_random + i_random, :, i_node + 1] = hand_velocity(
                socp_delay,
                x_prev[i_random * n_q : (i_random + 1) * n_q],
                x_prev[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q],
            )

    hand_pos_ref = np.zeros((2, n_shooting + 1))
    hand_vel_ref = np.zeros((2, n_shooting + 1))
    for i_node in range(n_shooting):
        hand_pos_ref[:, i_node] = variable_data["ref_fb_opt"][:2, i_node]
        hand_vel_ref[:, i_node] = variable_data["ref_fb_opt"][2:4, i_node]

    hand_initial_position, hand_final_position = get_target_position(socp_delay["model"])

    fig, axs = plt.subplots(3, 2)
    for i_simulation in range(n_simulations):
        axs[0, 0].plot(
            hand_pos_simulated[i_simulation, 0, :],
            hand_pos_simulated[i_simulation, 1, :],
            color=SOCP_DELAY_color,
            linewidth=0.5,
        )
        axs[1, 0].plot(
            np.linspace(0, final_time, n_shooting + 1),
            x_simulated[i_simulation, 0, :],
            color=SOCP_DELAY_color,
            linewidth=0.5,
        )
        axs[2, 0].plot(
            np.linspace(0, final_time, n_shooting + 1),
            x_simulated[i_simulation, 1, :],
            color=SOCP_DELAY_color,
            linewidth=0.5,
        )
        axs[0, 1].plot(
            np.linspace(0, final_time, n_shooting + 1),
            np.linalg.norm(hand_vel_simulated[i_simulation, :, :], axis=0),
            color=SOCP_DELAY_color,
            linewidth=0.5,
        )
        axs[1, 1].plot(
            np.linspace(0, final_time, n_shooting + 1),
            x_simulated[i_simulation, 2, :],
            color=SOCP_DELAY_color,
            linewidth=0.5,
        )
        axs[2, 1].plot(
            np.linspace(0, final_time, n_shooting + 1),
            x_simulated[i_simulation, 3, :],
            color=SOCP_DELAY_color,
            linewidth=0.5,
        )

    mean_q = np.mean(variable_data["q_opt"], axis=1)
    mean_qdot = np.mean(variable_data["q_opt"], axis=1)
    axs[0, 0].plot(hand_pos_ref[0, :], hand_pos_ref[1, :], color="k")
    axs[0, 0].plot(hand_initial_position[0], hand_initial_position[1], color="tab:green", marker="o", markersize=1)
    axs[0, 0].plot(hand_final_position[0], hand_final_position[1], color="tab:red", marker="o", markersize=1)
    axs[0, 0].set_xlabel("X [m]")
    axs[0, 0].set_ylabel("Y [m]")
    axs[0, 0].set_title("Hand position simulated")
    axs[1, 0].plot(np.linspace(0, final_time, n_shooting + 1), mean_q[0, :], color="k")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Shoulder angle [rad]")
    axs[2, 0].plot(np.linspace(0, final_time, n_shooting + 1), mean_q[1, :], color="k")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Elbow angle [rad]")
    axs[0, 1].plot(np.linspace(0, final_time, n_shooting + 1), np.linalg.norm(hand_vel_ref, axis=0), color="k")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Hand velocity [m/s]")
    axs[0, 1].set_title("Hand velocity simulated")
    axs[1, 1].plot(np.linspace(0, final_time, n_shooting + 1), mean_qdot[0, :], color="k")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Shoulder velocity [rad/s]")
    axs[2, 1].plot(np.linspace(0, final_time, n_shooting + 1), mean_qdot[1, :], color="k")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Elbow velocity [rad/s]")
    axs[0, 0].axis("equal")
    plt.tight_layout()
    save_path_fig = save_path_socp_delay.replace(".pkl", "_plot_hand_trajectories.png").replace(
        "/results/", "/figures/"
    )
    plt.savefig(save_path_fig)
    plt.show()
    # plt.close()


def plot_socp_delay(
    variable_data: dict[str, np.ndarray],
    socp_delay: dict[str, any],
    motor_noise_std: float,
    force_field_magnitude: float,
    save_path_socp_delay: str,
    n_simulations: int = 100,
):

    # TODO: see if force_field_magnitude is implemented correctly
    plot_states(variable_data, socp_delay, save_path_socp_delay)
    plot_controls(variable_data, socp_delay, save_path_socp_delay)
    plot_hand_trajectories(variable_data, socp_delay, n_simulations, save_path_socp_delay)
