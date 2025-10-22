import casadi as cas
import numpy as np

from ..models.arm_model import ArmModel


class DeterministicMultiArmModel(ArmModel):
    def __init__(
        self,
        motor_noise_std: float,
        n_random: int = 20,
        n_shooting: int = 50,
    ):
        ArmModel.__init__(
            self,
            n_random=n_random,
        )

        self.n_random = n_random
        self.n_shooting = n_shooting
        self.n_noises = self.nb_q

        # TODO: CHECK THE MOTOR NOISE MAGNITUDE (* dt)
        motor_noise_magnitude = cas.DM(np.array([motor_noise_std] * self.nb_q))
        self.motor_noise_magnitude = motor_noise_magnitude

    def collect_tau(self, qdot, tau, motor_noise_this_time):
        """
        Collect all tau components
        """
        tau_friction = -self.friction_coefficients @ qdot
        torques_computed = tau_friction + tau + motor_noise_this_time
        return torques_computed

    def dynamics(
        self,
        x_single,
        u_single,
        noise_single,
    ) -> cas.Function:
        """
        Variables:
        - q (2 x n_random, n_shooting + 1)
        - qdot (2 x n_random, n_shooting + 1)
        - tau (2, n_shooting)
        Noises:
        - motor_noise (6 x n_random, n_shooting)
        """

        # Collect variables
        tau = u_single[: self.nb_q]
        qddot = cas.MX.zeros(self.nb_q * self.n_random)
        noise_offset = 0
        for i_random in range(self.n_random):
            q_this_time = x_single[i_random * self.nb_q : (i_random + 1) * self.nb_q]
            qdot_this_time = x_single[self.q_offset + i_random * self.nb_q : self.q_offset + (i_random + 1) * self.nb_q]
            motor_noise_this_time = noise_single[noise_offset : noise_offset + self.nb_q]
            noise_offset += self.n_noises

            # Collect tau components
            torques_computed = self.collect_tau(qdot_this_time, tau, motor_noise_this_time)

            # Dynamics
            qddot[i_random * self.nb_q : (i_random + 1) * self.nb_q] = self.forward_dynamics(
                q_this_time, qdot_this_time, torques_computed
            )

        dxdt = cas.vertcat(x_single[self.q_offset :], qddot)
        return dxdt
