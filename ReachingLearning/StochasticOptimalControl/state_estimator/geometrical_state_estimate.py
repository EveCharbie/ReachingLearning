import casadi as cas
import numpy as np


def get_shoulder_angle_from_deltoid_anterior(deltoid_anterior_length):
    # eta
    deltoid_anterior_insertion_in_local = np.array([0.11, 0.015])
    eta = np.abs(cas.atan(deltoid_anterior_insertion_in_local[1] / deltoid_anterior_insertion_in_local[0]))

    # alpha
    deltoid_anterior_origin_in_global = np.array([-0.015, 0.04])
    alpha = np.abs(cas.atan(deltoid_anterior_origin_in_global[1] / deltoid_anterior_origin_in_global[0]))

    # phi
    deltoid_anterior_origin_norm = np.linalg.norm(deltoid_anterior_origin_in_global)
    deltoid_anterior_insertion_norm = np.linalg.norm(deltoid_anterior_insertion_in_local)
    phi = cas.acos(
        (deltoid_anterior_length**2 - deltoid_anterior_insertion_norm**2 - deltoid_anterior_origin_norm**2)
        / (-2 * deltoid_anterior_insertion_norm * deltoid_anterior_origin_norm)
    )

    theta_shoulder_ant = np.pi - phi - eta - alpha
    return theta_shoulder_ant


def get_shoulder_angle_from_deltoid_posterior(deltoid_posterior_length):
    # eta
    deltoid_posterior_insertion_in_local = np.array([0.11, -0.015])
    eta = np.abs(cas.atan(deltoid_posterior_insertion_in_local[1] / deltoid_posterior_insertion_in_local[0]))

    # alpha
    deltoid_posterior_origin_in_global = np.array([0.015, -0.04])
    alpha = np.abs(cas.atan(deltoid_posterior_origin_in_global[1] / deltoid_posterior_origin_in_global[0]))

    # phi
    deltoid_posterior_origin_norm = np.linalg.norm(deltoid_posterior_origin_in_global)
    deltoid_posterior_insertion_norm = np.linalg.norm(deltoid_posterior_insertion_in_local)
    phi = cas.acos(
        (deltoid_posterior_length**2 - deltoid_posterior_insertion_norm**2 - deltoid_posterior_origin_norm**2)
        / (-2 * deltoid_posterior_insertion_norm * deltoid_posterior_origin_norm)
    )

    theta_shoulder_post = phi + eta - alpha
    return theta_shoulder_post


def get_elbow_angle_from_brachialis(brachialis_length):
    # eta upper
    brachialis_origin_in_local = np.array([0.11, -0.025])
    upper_arm_length = 0.3
    brachialis_origin_from_elbow = upper_arm_length - brachialis_origin_in_local[0]
    eta_upper = np.abs(cas.atan(brachialis_origin_in_local[1] / brachialis_origin_from_elbow))

    # eta lower
    brachialis_insertion_in_local = np.array([0.025, 0.015])
    eta_lower = np.abs(cas.atan(brachialis_insertion_in_local[1] / brachialis_insertion_in_local[0]))

    # phi
    brachialis_origin_norm = np.sqrt(brachialis_origin_from_elbow**2 + brachialis_origin_in_local[1] ** 2)
    brachialis_insertion_norm = np.linalg.norm(brachialis_insertion_in_local)
    acos_val = ((brachialis_length**2 - brachialis_insertion_norm**2 - brachialis_origin_norm**2)
        / (-2 * brachialis_insertion_norm * brachialis_origin_norm))
    phi = cas.acos(cas.if_else(acos_val > 1, -1, acos_val))

    theta_elbow_bra = phi - eta_upper + eta_lower
    return theta_elbow_bra


def get_elbow_angle_from_lateral_triceps(triceps_lateral_length):
    # muscle segment lengths
    triceps_lateral_insertion_in_local = np.array([0.035, -0.015])
    vp1_in_local = np.array([-0.025, -0.025])
    vp2_in_local = np.array([0.025, -0.025])
    length_vp2_ins = np.linalg.norm(triceps_lateral_insertion_in_local - vp2_in_local)
    length_vp1_vp2 = np.linalg.norm(vp2_in_local - vp1_in_local)
    residual_length = triceps_lateral_length - length_vp2_ins - length_vp1_vp2

    # eta upper
    triceps_lateral_origin_in_local = np.array([0.11, -0.025])
    upper_arm_length = 0.3
    triceps_lateral_origin_from_elbow = upper_arm_length - triceps_lateral_origin_in_local[0]
    eta_upper = np.abs(cas.atan(triceps_lateral_origin_in_local[1] / triceps_lateral_origin_from_elbow))

    # eta vp2-insertion
    eta_1 = np.abs(cas.atan(vp2_in_local[1] / vp2_in_local[0]))

    # eta vp1-vp2
    eta_vp_vp = cas.acos(
        np.dot(vp1_in_local, vp2_in_local) / np.linalg.norm(vp1_in_local) / np.linalg.norm(vp2_in_local)
    )

    # phi
    upper_arm_length = 0.3
    triceps_lateral_origin_from_elbow = upper_arm_length - triceps_lateral_origin_in_local[0]
    triceps_lateral_origin_norm = np.sqrt(
        triceps_lateral_origin_from_elbow**2 + triceps_lateral_origin_in_local[1] ** 2
    )
    vp1_norm = np.linalg.norm(vp1_in_local)
    acos_val = ((residual_length**2 - triceps_lateral_origin_norm**2 - vp1_norm**2)
        / (-2 * triceps_lateral_origin_norm * vp1_norm))
    phi = cas.acos(cas.if_else(acos_val < -1, -1, acos_val))

    theta_elbow_tri_lat = 2 * np.pi - eta_1 - phi - eta_vp_vp - eta_upper
    return theta_elbow_tri_lat


def get_states_from_muscle_lengths(muscle_lengths):
    """
    Given mono-articular muscle lengths, it computes the corresponding joint angles (q) by averaging the estimates from
    the agonist/antagonist pair.
        - Shoulder angle from posterior deltoid and anterior deltoid
        - Elbow angle from brachialis and lateral triceps
    """
    # Shoulder
    theta_shoulder_ant = get_shoulder_angle_from_deltoid_anterior(muscle_lengths[2])
    theta_shoulder_post = get_shoulder_angle_from_deltoid_posterior(muscle_lengths[3])
    q1 = (theta_shoulder_ant + theta_shoulder_post) / 2
    if np.isnan(q1):
        print(f"Shoulder angle estimates : {theta_shoulder_ant}  {theta_shoulder_post}")

    # Elbow
    theta_elbow_bra = np.pi - get_elbow_angle_from_brachialis(muscle_lengths[0])
    theta_elbow_tri = np.pi - get_elbow_angle_from_lateral_triceps(muscle_lengths[1])
    q2 = (theta_elbow_bra + theta_elbow_tri) / 2
    if np.isnan(q2):
        print(f"Elbow angle estimates : {theta_elbow_bra}  {theta_elbow_tri}")
        theta_elbow_bra = np.pi - get_elbow_angle_from_brachialis(muscle_lengths[0])
        theta_elbow_tri = np.pi - get_elbow_angle_from_lateral_triceps(muscle_lengths[1])

    # print(f"Bra = {theta_elbow_bra - q2}, Tri = {theta_elbow_tri - q2}")
    return cas.vertcat(q1, q2)[:, 0]


def plot_state_estimation():
    import biorbd

    elbow_angles = [
            15,
            30,
            45,
            60,
            75,
            90,
            105,
            120,
            135,
            150,
        ]
    shoulder_angles = [15, 30, 45, 60, 75, 90]
    muscle_noise_magnitude = 0.0005

    biorbd_model = biorbd.Biorbd("../models/arm_model.bioMod")


    # Get the errors for one trial
    error_grid = np.zeros((len(shoulder_angles), len(elbow_angles), 2))
    for i_elbow, elbow_angle in enumerate(elbow_angles):
        for i_shoulder, shoulder_angle in enumerate(shoulder_angles):

            # Generate the data
            q = np.array([shoulder_angle * np.pi / 180, elbow_angle * np.pi / 180])
            muscle_lengths = biorbd_model.muscles.muscle_tendon_length(q)

            # Get the states estimated
            noise = np.random.normal(0, muscle_noise_magnitude, len(muscle_lengths))
            q_estimated = get_states_from_muscle_lengths(muscle_lengths + noise)
            error_grid[i_shoulder, i_elbow, :] = q - np.array(q_estimated).reshape((2,))


    # Get the mean error to make sure there are no nans
    error_grid_mean = np.zeros((len(shoulder_angles), len(elbow_angles), 2))
    for i_random in range(30):
        for i_elbow, elbow_angle in enumerate(elbow_angles):
            for i_shoulder, shoulder_angle in enumerate(shoulder_angles):

                # Generate the data
                q = np.array([shoulder_angle * np.pi / 180, elbow_angle * np.pi / 180])
                muscle_lengths = biorbd_model.muscles.muscle_tendon_length(q)

                # Get the states estimated
                noise = np.random.normal(0, muscle_noise_magnitude, len(muscle_lengths))
                q_estimated = get_states_from_muscle_lengths(muscle_lengths + noise)
                error_grid_mean[i_shoulder, i_elbow, :] += q - np.array(q_estimated).reshape((2,))
    error_grid_mean /= 30

    # Plot the error grid
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    im1 = ax[0].imshow(error_grid[:, :, 0] * 180 / np.pi, cmap='viridis', vmin=-7, vmax=7)
    # im1 = ax[0].imshow(error_grid_mean[:, :, 0] * 180 / np.pi, cmap='viridis', vmin=-7, vmax=7)
    ax[0].set_xticks(np.arange(len(elbow_angles)))
    ax[0].set_yticks(np.arange(len(shoulder_angles)))
    ax[0].set_xticklabels(elbow_angles)
    ax[0].set_yticklabels(shoulder_angles)
    ax[0].set_xlabel(r'Elbow angle [$^\circ$]')
    ax[0].set_ylabel(r'Shoulder angle [$^\circ$]')
    ax[0].set_title(r'Shoulder angle estimation error [$^\circ$]')
    ax[0].cbar = fig.colorbar(im1, ax=ax[0])
    ax[0].text(0, -2.5, f"Min error = {np.min(error_grid[:, :, 0]) * 180 / np.pi} deg")
    ax[0].text(0, -3.5, f"Max error = {np.max(error_grid[:, :, 0]) * 180 / np.pi} deg")
    ax[0].text(0, 8, f"Gaussian noise std={muscle_noise_magnitude} m")

    im2 = ax[1].imshow(error_grid[:, :, 1] * 180 / np.pi, cmap='viridis', vmin=-7, vmax=7)
    # im2 = ax[1].imshow(error_grid_mean[:, :, 1] * 180 / np.pi, cmap='viridis', vmin=-7, vmax=7)
    ax[1].set_xticks(np.arange(len(elbow_angles)))
    ax[1].set_yticks(np.arange(len(shoulder_angles)))
    ax[1].set_xticklabels(elbow_angles)
    ax[1].set_yticklabels(shoulder_angles)
    ax[1].set_xlabel(r'Elbow angle [$^\circ$]')
    ax[1].set_ylabel(r'Shoulder angle [$^\circ$]')
    ax[1].set_title(r'Elbow angle estimation error [$^\circ$]')
    ax[1].cbar = fig.colorbar(im1, ax=ax[1])
    ax[1].text(0, -2.5, f"Min error = {np.min(error_grid[:, :, 1]) * 180 / np.pi} deg")
    ax[1].text(0, -3.5, f"Max error = {np.max(error_grid[:, :, 1]) * 180 / np.pi} deg")
    ax[1].text(0, 8, f"Gaussian noise std={muscle_noise_magnitude} m")

    plt.tight_layout()
    plt.savefig("state_estimation_error.png")
    plt.show()


if __name__ == "__main__":
    plot_state_estimation()

