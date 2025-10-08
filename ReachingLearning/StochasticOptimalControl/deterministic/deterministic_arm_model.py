import casadi as cas
import numpy as np

from ..models.arm_model import ArmModel


class DeterministicArmModel(ArmModel):
    def __init__(
        self,
        force_field_magnitude: float = 0,
    ):
        ArmModel.__init__(
            self,
            force_field_magnitude=force_field_magnitude,
        )

    def dynamics(
        self,
        x_single,
        u_single,
    ) -> cas.Function:
        """
        Variables:
        - q (2, n_shooting + 1)
        - qdot (2, n_shooting + 1)
        - muscle (6, n_shooting)
        """

        # Collect variables
        q = x_single[0:2]
        qdot = x_single[2:4]
        muscle = u_single

        # Collect tau components
        muscles_tau = self.get_muscle_torque(q, qdot, muscle)
        tau_force_field = self.force_field(q, self.force_field_magnitude)
        tau_friction = - self.friction_coefficients @ qdot
        torques_computed = muscles_tau + tau_force_field + tau_friction

        # Dynamics
        qddot = self.forward_dynamics(q, qdot, torques_computed)
        dxdt = cas.vertcat(qdot, qddot)

        return dxdt

