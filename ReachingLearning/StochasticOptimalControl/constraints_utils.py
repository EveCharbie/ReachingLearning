import casadi as cas
import numpy as np

from .utils import ExampleType


def start_on_target(model, x_single: cas.MX) -> tuple[list[cas.MX], list[float], list[float]]:
    """
    Constraint to start on the target at the beginning of the movement
    """
    q = x_single[:2]
    ee_pos = model.end_effector_position(q)[:2]
    target = np.array([0.00000000, 0.27420000])
    g = [ee_pos - target]
    lbg = [0, 0]
    ubg = [0, 0]
    return g, lbg, ubg


def reach_target(model, x_single: cas.MX, example_type: ExampleType) -> tuple[list[cas.MX], list[float], list[float]]:
    """
    Constraint to reach the target at the end of the movement
    """
    q = x_single[:2]
    ee_pos = model.end_effector_position(q)[:2]
    target = np.array([0.00000000, 0.52730000])
    if example_type == ExampleType.BAR:
        g = [ee_pos[1] - target[1]]
        lbg = [0]
        ubg = [0]
    elif example_type == ExampleType.CIRCLE:
        g = [ee_pos - target]
        lbg = [0, 0]
        ubg = [0, 0]
    else:
        raise ValueError("example_type must be ExampleType.BAR or ExampleType.CIRCLE")

    return g, lbg, ubg
