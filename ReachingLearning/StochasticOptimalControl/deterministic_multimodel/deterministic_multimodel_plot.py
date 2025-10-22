import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from ..utils import get_target_position, get_dm_value
from ..plot_utils import set_columns_suptitles


ocp_multimodel_color = "#AC2594"


def hand_position(ocp_multimodel, q):
    hand_pos = get_dm_value(ocp_multimodel["model"].end_effector_position, [q])
    return np.reshape(hand_pos[:2], (2,))


def hand_velocity(ocp_multimodel, q, qdot):
    hand_velo = get_dm_value(ocp_multimodel["model"].end_effector_velocity, [q, qdot])
    return np.reshape(hand_velo[:2], (2,))

def plot_state_bounds(axs, time_vector, variable_data, n_shooting):
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

def plot_states(variable_data, ocp_multimodel, save_path_ocp_multimodel):

    n_shooting = ocp_multimodel["n_shooting"]
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
    for i_random in range(ocp_multimodel["model"].n_random):
        axs[0, 0].plot(time_vector, variable_data["q_opt"][0, i_random, :], ".", markersize=1, color=ocp_multimodel_color)
        axs[0, 1].plot(time_vector, variable_data["q_opt"][1, i_random, :], ".", markersize=1, color=ocp_multimodel_color)
        axs[1, 0].plot(
            time_vector, variable_data["qdot_opt"][0, i_random, :], ".", markersize=1, color=ocp_multimodel_color
        )
        axs[1, 1].plot(
            time_vector, variable_data["qdot_opt"][1, i_random, :], ".", markersize=1, color=ocp_multimodel_color
        )

    # Reintegration
    for i_random in range(ocp_multimodel["model"].n_random):
        axs[0, 0].plot(
            time_vector, variable_data["q_integrated"][0, i_random, :], "-", linewidth=0.5, color=ocp_multimodel_color
        )
        axs[0, 1].plot(
            time_vector, variable_data["q_integrated"][1, i_random, :], "-", linewidth=0.5, color=ocp_multimodel_color
        )
        axs[1, 0].plot(
            time_vector, variable_data["qdot_integrated"][0, i_random, :], "-", linewidth=0.5, color=ocp_multimodel_color
        )
        axs[1, 1].plot(
            time_vector, variable_data["qdot_integrated"][1, i_random, :], "-", linewidth=0.5, color=ocp_multimodel_color
        )

    # Bounds
    plot_state_bounds(axs, time_vector, variable_data, n_shooting)

    save_path_fig = save_path_ocp_multimodel.replace(".pkl", "_plot_states.png").replace("/results/", "/figures/")
    plt.savefig(save_path_fig)
    plt.show()
    # plt.close()

def plot_control_bounds(axs, time_vector, variable_data, n_shooting):
    for i_ax in range(2):
        axs[i_ax].fill_between(
            time_vector,
            np.ones((n_shooting,)) * -100,
            variable_data["lbtau"][i_ax, :],
            color="lightgrey",
        )
        axs[i_ax].fill_between(
            time_vector,
            variable_data["ubtau"][i_ax, :],
            np.ones((n_shooting,)) * 100,
            color="lightgrey",
        )
            
def plot_single_bounds(ax, time_vector, variable_data, n_shooting, bound_name):
    ax.fill_between(
        time_vector,
        np.ones((n_shooting,)) * np.min(variable_data["lb" + bound_name]) - 0.1,
        variable_data["lb" + bound_name][0, :],
        color="lightgrey",
    )
    ax.fill_between(
        time_vector,
        variable_data["ub" + bound_name][0, :],
        np.ones((n_shooting,)) * (np.max(variable_data["ub" + bound_name]) + 0.1),
        color="lightgrey",
    )
    ax.fill_between(
        time_vector,
        np.ones((n_shooting,)) * np.min(variable_data["lb" + bound_name]) - 0.1,
        variable_data["lb" + bound_name][1, :],
        color="lightgrey",
    )
    ax.fill_between(
        time_vector,
        variable_data["ub" + bound_name][1, :],
        np.ones((n_shooting,)) * (np.max(variable_data["ub" + bound_name]) + 0.1),
        color="lightgrey",
    )

def plot_controls(variable_data, ocp_multimodel, save_path_ocp_multimodel):

    n_shooting = ocp_multimodel["n_shooting"]
    time_vector = variable_data["time_vector"][:-1]

    # Set up the plots
    fig, axs = plt.subplots(1, 2, figsize=(6, 6))
    axs[0].set_xlabel("Time [s]")
    axs[1].set_xlabel("Time [s]")

    # Optimization variables
    axs[0].plot(time_vector, variable_data["tau_opt"][0, :], ".", markersize=1, color=ocp_multimodel_color)
    axs[1].plot(time_vector, variable_data["tau_opt"][1, :], ".", markersize=1, color=ocp_multimodel_color)

    # Bounds
    plot_control_bounds(axs, time_vector, variable_data, n_shooting)
    axs[0].set_ylim([np.min(variable_data["lbtau"]) - 0.1, np.max(variable_data["ubtau"]) + 0.1])
    axs[1].set_ylim([np.min(variable_data["lbtau"]) - 0.1, np.max(variable_data["ubtau"]) + 0.1])
    axs[0].set_title("Shoulder torque")
    axs[0].set_ylabel("Torque [Nm]")
    axs[1].set_title("Elbow torque")
    axs[1].set_ylabel("Torque [Nm]")

    plt.tight_layout()
    save_path_fig = save_path_ocp_multimodel.replace(".pkl", "_plot_controls.png").replace("/results/", "/figures/")
    plt.savefig(save_path_fig)
    plt.show()
    # plt.close()


def plot_hand_trajectories(variable_data, ocp_multimodel, n_simulations, save_path_ocp_multimodel):

    n_shooting = ocp_multimodel["n_shooting"]
    final_time = ocp_multimodel["final_time"]
    n_random = ocp_multimodel["model"].n_random
    q_offset = ocp_multimodel["model"].q_offset
    n_q = ocp_multimodel["model"].nb_q
    n_noises = ocp_multimodel["model"].nb_q * n_random
    noise_magnitude = np.array(
                np.array(ocp_multimodel["model"].motor_noise_magnitude)
                .reshape(
                    -1,
                )
                .tolist()
                * n_random
            )

    # Reintegrate the solution with noise
    x_simulated = np.zeros((n_simulations * n_random, 2 * n_q, n_shooting + 1))
    hand_pos_simulated = np.zeros((n_simulations * n_random, 2, n_shooting + 1))
    hand_vel_simulated = np.zeros((n_simulations * n_random, 2, n_shooting + 1))
    for i_simulation in range(n_simulations):
        print(f"Running ocp_multimodel noised simulation {i_simulation}")
        np.random.seed(i_simulation)
        for i_random in range(n_random):
            i_simulation_total = i_simulation * n_random + i_random
            x_simulated[i_simulation_total, :n_q, 0] = variable_data["x_opt"][
                i_random * n_q : (i_random + 1) * n_q, 0
            ]
            x_simulated[i_simulation_total, n_q:, 0] = variable_data["x_opt"][
                n_q * n_random + i_random * n_q : n_q * n_random + (i_random + 1) * n_q, 0
            ]
        for i_node in range(n_shooting):
            x_prev = np.zeros((n_q * 2 * n_random))
            for i_random in range(n_random):
                i_simulation_total = i_simulation * n_random + i_random
                x_prev[i_random * n_q : (i_random + 1) * n_q] = x_simulated[
                    i_simulation_total, :n_q, i_node
                ]
                x_prev[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q] = x_simulated[
                    i_simulation_total,
                    n_q:,
                    i_node,
                ]
            u_this_time = variable_data["u_opt"][:, i_node]
            noise_this_time = np.random.normal(0, noise_magnitude, n_noises)
            x_next = ocp_multimodel["integration_func"](x_prev, u_this_time, noise_this_time)
            for i_random in range(n_random):
                i_simulation_total = i_simulation * n_random + i_random
                x_simulated[i_simulation_total, :n_q, i_node + 1] = np.reshape(
                    x_next[i_random * n_q : (i_random + 1) * n_q, 0], (-1,)
                )
                x_simulated[i_simulation_total, n_q:, i_node + 1] = np.reshape(
                    x_next[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q, 0], (-1,)
                )

                hand_pos_simulated[i_simulation_total, :, i_node] = hand_position(
                    ocp_multimodel, x_prev[i_random * n_q : (i_random + 1) * n_q]
                )
                hand_vel_simulated[i_simulation_total, :, i_node] = hand_velocity(
                    ocp_multimodel,
                    x_prev[i_random * n_q : (i_random + 1) * n_q],
                    x_prev[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q],
                )

        # Final point
        x_prev = np.zeros((n_q * 2 * n_random))
        for i_random in range(n_random):
            i_simulation_total = i_simulation * n_random + i_random
            x_prev[i_random * n_q : (i_random + 1) * n_q] = x_simulated[
                i_simulation_total, :n_q, i_node + 1
            ]
            x_prev[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q] = x_simulated[
                i_simulation_total,
                n_q:,
                i_node + 1,
            ]
        for i_random in range(n_random):
            hand_pos_simulated[i_simulation_total, :, i_node + 1] = hand_position(
                ocp_multimodel, x_prev[i_random * n_q : (i_random + 1) * n_q]
            )
            hand_vel_simulated[i_simulation_total, :, i_node + 1] = hand_velocity(
                ocp_multimodel,
                x_prev[i_random * n_q : (i_random + 1) * n_q],
                x_prev[q_offset + i_random * n_q : q_offset + (i_random + 1) * n_q],
            )

    hand_initial_position, hand_final_position = get_target_position(ocp_multimodel["model"])

    fig, axs = plt.subplots(3, 2)
    for i_simulation in range(n_simulations):
        for i_random in range(n_random):
            i_simulation_total = i_simulation * n_random + i_random
            axs[0, 0].plot(
                hand_pos_simulated[i_simulation_total, 0, :],
                hand_pos_simulated[i_simulation_total, 1, :],
                color=ocp_multimodel_color,
                linewidth=0.5,
            )
            axs[1, 0].plot(
                np.linspace(0, final_time, n_shooting + 1),
                x_simulated[i_simulation_total, 0, :],
                color=ocp_multimodel_color,
                linewidth=0.5,
            )
            axs[2, 0].plot(
                np.linspace(0, final_time, n_shooting + 1),
                x_simulated[i_simulation_total, 1, :],
                color=ocp_multimodel_color,
                linewidth=0.5,
            )
            axs[0, 1].plot(
                np.linspace(0, final_time, n_shooting + 1),
                np.linalg.norm(hand_vel_simulated[i_simulation_total, :, :], axis=0),
                color=ocp_multimodel_color,
                linewidth=0.5,
            )
            axs[1, 1].plot(
                np.linspace(0, final_time, n_shooting + 1),
                x_simulated[i_simulation_total, 2, :],
                color=ocp_multimodel_color,
                linewidth=0.5,
            )
            axs[2, 1].plot(
                np.linspace(0, final_time, n_shooting + 1),
                x_simulated[i_simulation_total, 3, :],
                color=ocp_multimodel_color,
                linewidth=0.5,
            )

    mean_q = np.mean(variable_data["q_opt"], axis=1)
    mean_qdot = np.mean(variable_data["qdot_opt"], axis=1)
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
    save_path_fig = save_path_ocp_multimodel.replace(".pkl", "_plot_hand_trajectories.png").replace(
        "/results/", "/figures/"
    )
    plt.savefig(save_path_fig)
    plt.show()
    # plt.close()


def plot_ocp_multimodel(
    variable_data: dict[str, np.ndarray],
    ocp_multimodel: dict[str, any],
    save_path_ocp_multimodel: str,
    n_simulations: int = 100,
):

    plot_states(variable_data, ocp_multimodel, save_path_ocp_multimodel)
    plot_controls(variable_data, ocp_multimodel, save_path_ocp_multimodel)
    plot_hand_trajectories(variable_data, ocp_multimodel, n_simulations, save_path_ocp_multimodel)
