import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from ..utils import get_target_position, get_dm_value


OCP_color = "#5DC962"


def hand_position(ocp, q):
    hand_pos = get_dm_value(ocp["model"].end_effector_position, [q])
    return np.reshape(hand_pos[:2], (2,))


def hand_velocity(ocp, q, qdot):
    hand_velo = get_dm_value(ocp["model"].end_effector_velocity, [q, qdot])
    return np.reshape(hand_velo[:2], (2,))


def plot_variables(variable_data, ocp, save_path_ocp):

    n_shooting = ocp["n_shooting"]

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
    axs[0, 0].set_xlim([0, variable_data["time_vector"][-1]])
    axs[0, 1].set_xlim([0, variable_data["time_vector"][-1]])
    axs[1, 0].set_xlim([0, variable_data["time_vector"][-1]])
    axs[1, 1].set_xlim([0, variable_data["time_vector"][-1]])

    # Optimization variables
    axs[0, 0].plot(variable_data["time_vector"], variable_data["q_opt"][0, :], ".", color=OCP_color)
    axs[0, 1].plot(variable_data["time_vector"], variable_data["q_opt"][1, :], ".", color=OCP_color)
    axs[1, 0].plot(variable_data["time_vector"], variable_data["qdot_opt"][0, :], ".", color=OCP_color)
    axs[1, 1].plot(variable_data["time_vector"], variable_data["qdot_opt"][1, :], ".", color=OCP_color)

    # Reintegration
    axs[0, 0].plot(variable_data["time_vector"], variable_data["q_integrated"][0, :], "-", linewidth=2, color=OCP_color)
    axs[0, 1].plot(variable_data["time_vector"], variable_data["q_integrated"][1, :], "-", linewidth=2, color=OCP_color)
    axs[1, 0].plot(
        variable_data["time_vector"], variable_data["qdot_integrated"][0, :], "-", linewidth=2, color=OCP_color
    )
    axs[1, 1].plot(
        variable_data["time_vector"], variable_data["qdot_integrated"][1, :], "-", linewidth=2, color=OCP_color
    )

    # Bounds
    axs[0, 0].fill_between(
        variable_data["time_vector"], np.ones((n_shooting + 1,)) * -10, variable_data["lbq"][0, :], color="lightgrey"
    )
    axs[0, 0].fill_between(
        variable_data["time_vector"], variable_data["ubq"][0, :], np.ones((n_shooting + 1,)) * 10, color="lightgrey"
    )
    axs[0, 1].fill_between(
        variable_data["time_vector"], np.ones((n_shooting + 1,)) * -10, variable_data["lbq"][1, :], color="lightgrey"
    )
    axs[0, 1].fill_between(
        variable_data["time_vector"], variable_data["ubq"][1, :], np.ones((n_shooting + 1,)) * 10, color="lightgrey"
    )
    axs[1, 0].fill_between(
        variable_data["time_vector"],
        np.ones((n_shooting + 1,)) * -100,
        variable_data["lbqdot"][0, :],
        color="lightgrey",
    )
    axs[1, 0].fill_between(
        variable_data["time_vector"], variable_data["ubqdot"][0, :], np.ones((n_shooting + 1,)) * 100, color="lightgrey"
    )
    axs[1, 1].fill_between(
        variable_data["time_vector"],
        np.ones((n_shooting + 1,)) * -100,
        variable_data["lbqdot"][1, :],
        color="lightgrey",
    )
    axs[1, 1].fill_between(
        variable_data["time_vector"], variable_data["ubqdot"][1, :], np.ones((n_shooting + 1,)) * 100, color="lightgrey"
    )

    plt.tight_layout()
    save_path_fig = save_path_ocp.replace(".pkl", "_plot_variables.png").replace("/results/", "/figures/")
    plt.savefig(save_path_fig)
    plt.show()
    # plt.close()


def plot_hand_trajectories(variable_data, ocp, n_simulations, motor_noise_std, save_path_ocp):

    n_shooting = ocp["n_shooting"]
    final_time = ocp["final_time"]

    # Reintegrate the solution with noise (for comparison with stochastic OCP)
    q_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
    qdot_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
    hand_pos_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
    hand_vel_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
    for i_simulation in range(n_simulations):
        print(f"Running OCP noised simulation {i_simulation}")
        np.random.seed(i_simulation)
        motor_noise = np.random.normal(0, motor_noise_std, (6, n_shooting + 1))
        q_simulated[i_simulation, :, 0] = variable_data["q_opt"][:, 0]
        qdot_simulated[i_simulation, :, 0] = variable_data["qdot_opt"][:, 0]
        for i_node in range(n_shooting):
            x_prev = cas.vertcat(
                q_simulated[i_simulation, :, i_node],
                qdot_simulated[i_simulation, :, i_node],
            )
            x_next = ocp["integration_func"](x_prev, variable_data["muscle_opt"][:, i_node] + motor_noise[:, i_node])
            q_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[:2, 0], (2,))
            qdot_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[2:4, 0], (2,))

            hand_pos_simulated[i_simulation, :, i_node] = hand_position(ocp, x_prev[:2])
            hand_vel_simulated[i_simulation, :, i_node] = hand_velocity(ocp, x_prev[:2], x_prev[2:4])

        x_prev = cas.vertcat(
            q_simulated[i_simulation, :, i_node + 1],
            qdot_simulated[i_simulation, :, i_node + 1],
        )
        hand_pos_simulated[i_simulation, :, i_node + 1] = hand_position(ocp, x_prev[:2])
        hand_vel_simulated[i_simulation, :, i_node + 1] = hand_velocity(ocp, x_prev[:2], x_prev[2:4])

    hand_pos_without_noise = np.zeros((2, n_shooting + 1))
    hand_vel_without_noise = np.zeros((2, n_shooting + 1))
    for i_node in range(n_shooting + 1):
        hand_pos_without_noise[:, i_node] = hand_position(ocp, variable_data["q_opt"][:, i_node])
        hand_vel_without_noise[:, i_node] = hand_velocity(
            ocp, variable_data["q_opt"][:, i_node], variable_data["qdot_opt"][:, i_node]
        )

    hand_initial_position, hand_final_position = get_target_position(ocp["model"])

    fig, axs = plt.subplots(3, 2)
    for i_simulation in range(n_simulations):
        axs[0, 0].plot(
            hand_pos_simulated[i_simulation, 0, :],
            hand_pos_simulated[i_simulation, 1, :],
            color=OCP_color,
            linewidth=0.5,
        )
        axs[1, 0].plot(
            np.linspace(0, final_time, n_shooting + 1),
            q_simulated[i_simulation, 0, :],
            color=OCP_color,
            linewidth=0.5,
        )
        axs[2, 0].plot(
            np.linspace(0, final_time, n_shooting + 1),
            q_simulated[i_simulation, 1, :],
            color=OCP_color,
            linewidth=0.5,
        )
        axs[0, 1].plot(
            np.linspace(0, final_time, n_shooting + 1),
            np.linalg.norm(hand_vel_simulated[i_simulation, :, :], axis=0),
            color=OCP_color,
            linewidth=0.5,
        )
        axs[1, 1].plot(
            np.linspace(0, final_time, n_shooting + 1),
            qdot_simulated[i_simulation, 0, :],
            color=OCP_color,
            linewidth=0.5,
        )
        axs[2, 1].plot(
            np.linspace(0, final_time, n_shooting + 1),
            qdot_simulated[i_simulation, 1, :],
            color=OCP_color,
            linewidth=0.5,
        )

    axs[0, 0].plot(hand_pos_without_noise[0, :], hand_pos_without_noise[1, :], color="k")
    axs[0, 0].plot(hand_initial_position[0], hand_initial_position[1], color="tab:green", marker="o", markersize=2)
    axs[0, 0].plot(hand_final_position[0], hand_final_position[1], color="tab:red", marker="o", markersize=2)
    axs[0, 0].set_xlabel("X [m]")
    axs[0, 0].set_ylabel("Y [m]")
    axs[0, 0].set_title("Hand position simulated")
    axs[1, 0].plot(np.linspace(0, final_time, n_shooting + 1), variable_data["q_opt"][0, :], color="k")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Shoulder angle [rad]")
    axs[2, 0].plot(np.linspace(0, final_time, n_shooting + 1), variable_data["q_opt"][1, :], color="k")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Elbow angle [rad]")
    axs[0, 1].plot(
        np.linspace(0, final_time, n_shooting + 1), np.linalg.norm(hand_vel_without_noise, axis=0), color="k"
    )
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Hand velocity [m/s]")
    axs[0, 1].set_title("Hand velocity simulated")
    axs[1, 1].plot(np.linspace(0, final_time, n_shooting + 1), variable_data["qdot_opt"][0, :], color="k")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Shoulder velocity [rad/s]")
    axs[2, 1].plot(np.linspace(0, final_time, n_shooting + 1), variable_data["qdot_opt"][1, :], color="k")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Elbow velocity [rad/s]")
    axs[0, 0].axis("equal")
    plt.tight_layout()
    save_path_fig = save_path_ocp.replace(".pkl", "_plot_hand_trajectories.png")
    plt.savefig(save_path_fig)
    plt.show()
    # plt.close()


def plot_ocp(
    variable_data: dict[str, np.ndarray],
    ocp: dict[str, any],
    motor_noise_std: float,
    force_field_magnitude: float,
    save_path_ocp: str,
    n_simulations: int = 100,
):

    # TODO: see if force_field_magnitude is implemented correctly
    plot_variables(variable_data, ocp, save_path_ocp)
    plot_hand_trajectories(variable_data, ocp, n_simulations, motor_noise_std, save_path_ocp)
