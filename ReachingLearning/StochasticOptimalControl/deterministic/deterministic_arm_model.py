import casadi as cas
import numpy as np

from ..models.arm_model import ArmModel


class DeterministicArmModel(ArmModel):
    def __init__(
        self,
        force_field_magnitude: float = 0,
        n_shooting: int = 50,
        forward_dynamics_func: cas.Function = None,
        muscle_driven: bool = True,
    ):
        ArmModel.__init__(
            self,
            force_field_magnitude=force_field_magnitude,
        )
        self.forward_dynamics_func = forward_dynamics_func
        self.n_shooting = n_shooting
        self.force_field_magnitude = force_field_magnitude
        self.muscle_driven = muscle_driven

    def real_dynamics(
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

        # Collect tau components
        if self.muscle_driven:
            muscle = u_single
            muscles_tau = self.get_muscle_torque(q, qdot, muscle)
        else:
            muscles_tau = u_single
        tau_force_field = self.force_field(q, self.force_field_magnitude)
        tau_friction = -self.friction_coefficients @ qdot
        torques_computed = muscles_tau + tau_force_field + tau_friction

        # Dynamics
        qddot = self.forward_dynamics(q, qdot, torques_computed)
        dxdt = cas.vertcat(qdot, qddot)

        return dxdt

    def dynamics(
        self,
        x_single,
        u_single,
    ) -> cas.Function:
        if self.forward_dynamics_func is None:
            # Biorbd dynamics
            return self.real_dynamics(x_single, u_single)
        else:
            # Dynamics learned in LearningInternalDynamics
            return self.forward_dynamics_func(x_single, u_single)
