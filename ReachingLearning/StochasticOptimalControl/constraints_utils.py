import casadi as cas
import numpy as np

from .utils import ExampleType
from .penalty_utils import (
    get_end_effector_for_all_random,
    get_end_effector_position_for_all_random,
    get_end_effector_velocity_for_all_random,
)
from ..constants import TARGET_START_VAN_WOUWE, TARGET_END_VAN_WOUWE


def start_on_target(model, x_single: cas.MX, target_start: np.ndarray=None) -> tuple[list[cas.MX], list[float], list[float]]:
    """
    Constraint to start on the target at the beginning of the movement
    """
    target = TARGET_START_VAN_WOUWE if target_start is None else target_start
    q = x_single[:2]
    ee_pos = model.end_effector_position(q)[:2]
    g = [ee_pos - target]
    lbg = [0, 0]
    ubg = [0, 0]
    return g, lbg, ubg


def mean_start_on_target(model, x_single: cas.MX, target_start: np.ndarray = None) -> tuple[list[cas.MX], list[float], list[float]]:
    """
    Constraint to impose that the mean trajectory reaches the target at the end of the movement
    """
    target = TARGET_START_VAN_WOUWE if target_start is None else target_start
    nb_random = model.n_random
    ee_pos = get_end_effector_position_for_all_random(model, x_single)
    ee_pos_mean = cas.sum2(ee_pos) / nb_random
    g = [ee_pos_mean - target]
    lbg = [0, 0]
    ubg = [0, 0]
    return g, lbg, ubg


def reach_target(model, x_single: cas.MX, example_type: ExampleType, target_end: np.ndarray=None) -> tuple[list[cas.MX], list[float], list[float]]:
    """
    Constraint to reach the target at the end of the movement
    """
    target_end = TARGET_END_VAN_WOUWE if target_end is None else target_end
    q = x_single[:2]
    ee_pos = model.end_effector_position(q)[:2]
    if example_type == ExampleType.BAR:
        g = [ee_pos[1] - target_end[1]]
        lbg = [0]
        ubg = [0]
    elif example_type == ExampleType.CIRCLE:
        g = [ee_pos - target_end]
        lbg = [0, 0]
        ubg = [0, 0]
    else:
        raise ValueError("example_type must be ExampleType.BAR or ExampleType.CIRCLE")

    return g, lbg, ubg


def mean_reach_target(
    model, x_single: cas.MX, example_type: ExampleType, target_end: np.ndarray = None
) -> tuple[list[cas.MX], list[float], list[float]]:
    """
    Constraint to impose that the mean trajectory reaches the target at the end of the movement
    """
    target_end = TARGET_END_VAN_WOUWE if target_end is None else target_end
    nb_random = model.n_random
    ee_pos = get_end_effector_position_for_all_random(model, x_single)
    ee_pos_mean = cas.sum2(ee_pos) / nb_random
    if example_type == ExampleType.BAR:
        g = [ee_pos_mean[1] - target_end[1]]
        lbg = [0]
        ubg = [0]
    elif example_type == ExampleType.CIRCLE:
        g = [ee_pos_mean - target_end]
        lbg = [0, 0]
        ubg = [0, 0]
    else:
        raise ValueError("example_type must be ExampleType.BAR or ExampleType.CIRCLE")

    return g, lbg, ubg


def mean_end_effector_velocity(model, x_single: cas.MX) -> list[cas.MX]:
    """
    Constraint to impose that the mean hand velocity is null at the end of the movement
    """
    nb_random = model.n_random
    ee_velo = get_end_effector_velocity_for_all_random(model, x_single)
    ee_pos_mean = cas.sum2(ee_velo) / nb_random
    return [ee_pos_mean]


def ref_equals_mean_ref(model, x_single, u_single) -> list[cas.MX]:
    """
    Constraint to impose that the feedback reference is equal to the mean feedback function value
    """
    nb_random = model.n_random
    ref_fb = u_single[model.ref_fb_indices]
    ee_pos, ee_vel = get_end_effector_for_all_random(model, x_single)
    ee_pos_mean = cas.sum2(ee_pos) / nb_random
    ee_vel_mean = cas.sum2(ee_vel) / nb_random
    g = [ref_fb - cas.vertcat(ee_pos_mean, ee_vel_mean)]
    return g


def residual_tau_equals_zero(model, u_single) -> list[cas.MX]:
    """
    Constraint to impose that the residual torque are null at convergence
    """
    offset = model.nb_muscles + model.nb_q * model.n_references + model.n_references
    tau = u_single[offset : offset + model.nb_q]
    return [tau]
