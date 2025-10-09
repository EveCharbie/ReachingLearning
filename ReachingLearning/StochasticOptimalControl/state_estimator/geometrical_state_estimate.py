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
    phi = cas.acos((deltoid_anterior_length ** 2 - deltoid_anterior_insertion_norm ** 2 - deltoid_anterior_origin_norm ** 2) / (-2 * deltoid_anterior_insertion_norm * deltoid_anterior_origin_norm))

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
    phi = cas.acos((deltoid_posterior_length ** 2 - deltoid_posterior_insertion_norm ** 2 - deltoid_posterior_origin_norm ** 2) / (-2 * deltoid_posterior_insertion_norm * deltoid_posterior_origin_norm))

    theta_shoulder_post = phi + eta - alpha
    return theta_shoulder_post


def get_elbow_angle_from_brachialis(brachialis_length):
    # eta upper
    brachialis_origin_in_local = np.array([0.11, 0.03])
    upper_arm_length = 0.3
    brachialis_origin_from_elbow = upper_arm_length - brachialis_origin_in_local[0]
    eta_upper = np.abs(cas.atan(brachialis_origin_in_local[1] / brachialis_origin_from_elbow))

    # eta lower
    brachialis_insertion_in_local = np.array([0.025, 0.015])
    eta_lower = np.abs(cas.atan(brachialis_insertion_in_local[1] / brachialis_insertion_in_local[0]))

    # phi
    brachialis_origin_norm = np.sqrt(brachialis_origin_from_elbow ** 2 + brachialis_origin_in_local[1] ** 2)
    brachialis_insertion_norm = np.linalg.norm(brachialis_insertion_in_local)
    phi = cas.acos((brachialis_length ** 2 - brachialis_insertion_norm ** 2 - brachialis_origin_norm ** 2) / (-2 * brachialis_insertion_norm * brachialis_origin_norm))

    # Handle the cosine ambiguity
    # phi = cas.fmod(phi, np.pi/2)
    phi = cas.if_else(brachialis_length ** 2 < (brachialis_insertion_norm ** 2 + brachialis_origin_norm ** 2), np.pi - phi, phi)

    theta_elbow_bra = phi + eta_upper + eta_lower
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

    # eta lower
    # eta_lower = cas.atan(triceps_lateral_insertion_in_local[1] / triceps_lateral_insertion_in_local[0])

    # eta vp2-insertion
    eta_1 = np.abs(cas.atan(vp2_in_local[1] / vp2_in_local[0]))

    # eta vp1-vp2
    eta_vp_vp = cas.acos(np.dot(vp1_in_local, vp2_in_local) / np.linalg.norm(vp1_in_local) / np.linalg.norm(vp2_in_local))

    # Handle the cosine ambiguity
    # eta_vp_vp = cas.fmod(eta_vp_vp, np.pi/2)
    eta_vp_vp = cas.if_else(length_vp1_vp2 ** 2 > (np.linalg.norm(vp1_in_local) ** 2 + np.linalg.norm(vp2_in_local) ** 2), np.pi - eta_vp_vp, eta_vp_vp)

    # phi
    triceps_lateral_origin_in_local = np.array([0.11, -0.025])
    upper_arm_length = 0.3
    triceps_lateral_origin_from_elbow = upper_arm_length - triceps_lateral_origin_in_local[0]
    triceps_lateral_origin_norm = np.sqrt(triceps_lateral_origin_from_elbow ** 2 + triceps_lateral_origin_in_local[1] ** 2)
    vp1_norm = np.linalg.norm(vp1_in_local)
    phi = cas.acos((residual_length ** 2 - triceps_lateral_origin_norm ** 2 - vp1_norm ** 2) / (-2 * triceps_lateral_origin_norm * vp1_norm))

    # Handle the cosine ambiguity
    # phi = cas.fmod(phi, np.pi/2)
    phi = cas.if_else(residual_length ** 2 > (triceps_lateral_origin_norm ** 2 + vp1_norm ** 2), np.pi - phi, phi)

    theta_elbow_tri_lat = 2*np.pi - eta_1 - phi - eta_vp_vp - eta_upper
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

    # Elbow
    theta_elbow_bra = get_elbow_angle_from_brachialis(muscle_lengths[0])
    theta_elbow_tri = get_elbow_angle_from_lateral_triceps(muscle_lengths[1])
    q2 = np.pi - ((theta_elbow_bra + theta_elbow_tri) / 2)

    return cas.vertcat(q1, q2)[:, 0]


