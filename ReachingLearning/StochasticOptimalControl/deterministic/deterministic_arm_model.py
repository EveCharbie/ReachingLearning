import casadi as cas
import numpy as np

from ..models.arm_model import ArmModel

    
def skip(*args, **kwargs):
    pass

class DeterministicArmModel(ArmModel):
    def __init__(
        self,
        force_field_magnitude: float = 0,
    ):
        ArmModel.__init__(
            self,
            sensory_noise_magnitude=np.zeros((4, 1)),
            motor_noise_magnitude=np.zeros((6, 1)),
            force_field_magnitude=force_field_magnitude,
        )
        self.sensory_reference = self.end_effector_pos_velo

    def dynamics(
            self,
            x,
            u,
    ) -> cas.Function:
        """
        OCP dynamics
            # muscle,
            # fb_ref,
            # fb_gains,
            # ff_ref,
            # ff_gains,
            # sensory_noise,
            # motor_noise,
        """

        q = x[0:2]
        qdot = x[2:4]
        muscle = u

        muscles_tau = self.get_muscle_torque(q, qdot, muscle)

        tau_force_field = self.force_field(q, self.force_field_magnitude)

        torques_computed = muscles_tau + tau_force_field

        dq_computed = qdot

        a1 = self.I1 + self.I2 + self.m2 * self.l1**2
        a2 = self.m2 * self.l1 * self.lc2
        a3 = self.I2

        theta_elbow = q[1]
        dtheta_shoulder = qdot[0]
        dtheta_elbow = qdot[1]

        cx = type(theta_elbow)
        mass_matrix = cx(2, 2)
        mass_matrix[0, 0] = a1 + 2 * a2 * cas.cos(theta_elbow)
        mass_matrix[0, 1] = a3 + a2 * cas.cos(theta_elbow)
        mass_matrix[1, 0] = a3 + a2 * cas.cos(theta_elbow)
        mass_matrix[1, 1] = a3

        nleffects = cx(2, 1)
        nleffects[0] = a2 * cas.sin(theta_elbow) * (-dtheta_elbow * (2 * dtheta_shoulder + dtheta_elbow))
        nleffects[1] = a2 * cas.sin(theta_elbow) * dtheta_shoulder**2

        friction = self.friction_coefficients

        dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - nleffects - friction @ qdot)
        dxdt = cas.vertcat(dq_computed, dqdot_computed)

        return dxdt


