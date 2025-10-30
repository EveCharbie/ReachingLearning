import casadi as cas
import numpy as np

from ..models.arm_model import ArmModel


class StochasticBasicArmModel(ArmModel):
    def __init__(
        self,
        motor_noise_std: float,
        wPq_std: float,
        wPqdot_std: float,
        force_field_magnitude: float = 0,
        n_random: int = 20,
        n_shooting: int = 50,
    ):
        ArmModel.__init__(
            self,
            force_field_magnitude=force_field_magnitude,
            n_random=n_random,
        )

        self.n_random = n_random
        self.n_shooting = n_shooting
        self.force_field_magnitude = force_field_magnitude
        self.n_references = 4  # 2 hand position + 2 hand velocity
        self.n_noises = self.n_references + self.nb_q

        motor_noise_magnitude = cas.DM(np.array([motor_noise_std] * self.nb_q))
        hand_sensory_noise_magnitude = cas.DM(
            np.array(
                [
                    wPq_std,  # Hand position
                    wPq_std,
                    wPqdot_std,  # Hand velocity
                    wPqdot_std,
                ]
            )
        )
        self.motor_noise_magnitude = motor_noise_magnitude
        self.hand_sensory_noise_magnitude = hand_sensory_noise_magnitude

        self.matrix_shape_k_fb = (self.nb_q, self.n_references)

    def sensory_reference(self, q, qdot, sensory_noise):
        """
        Sensory feedback: hand position and velocity
        """
        ee_pos = self.end_effector_position(q)
        ee_vel = self.end_effector_velocity(q, qdot)
        return cas.vertcat(ee_pos, ee_vel) + sensory_noise

    def collect_tau(self, q, qdot, muscle_activations, k_fb, ref_fb, tau, motor_noise_this_time, sensory_noise_this_time):
        """
        Collect all tau components

        Note: that the following line compromises convergence :(
        `muscles_tau = self.get_muscle_torque(q, qdot, muscle_activations + motor_noise_this_time)`
        So we add the noise on tau instead
        """
        muscles_tau = self.get_muscle_torque(q, qdot, muscle_activations)
        tau_force_field = self.force_field(q, self.force_field_magnitude)
        tau_fb = k_fb @ (self.sensory_reference(q, qdot, sensory_noise_this_time) - ref_fb)
        tau_friction = -self.friction_coefficients @ qdot
        torques_computed = muscles_tau + tau_force_field + tau_fb + tau_friction + tau + motor_noise_this_time
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
        - muscle (6, n_shooting)
        - k_fb (4 x 6, n_shooting)
        - ref_fb (4, n_shooting)
        - tau (2, n_shooting)
        Noises:
        - motor_noise (2 x n_random, n_shooting)
        - sensory_noise (4 x n_random, n_shooting)
        """

        # Collect variables
        muscle_activations = u_single[: self.nb_muscles]
        muscle_offset = self.nb_muscles
        k_fb = self.reshape_vector_to_matrix(
            u_single[muscle_offset : muscle_offset + self.n_references * self.nb_q], self.matrix_shape_k_fb
        )
        k_fb_offset = muscle_offset + self.n_references * self.nb_q
        ref_fb = u_single[k_fb_offset : k_fb_offset + self.n_references]
        ref_offset = k_fb_offset + self.n_references
        tau = u_single[ref_offset : ref_offset + self.nb_q]
        qddot = cas.MX.zeros(self.nb_q * self.n_random)
        noise_offset = 0
        for i_random in range(self.n_random):
            q_this_time = x_single[i_random * self.nb_q : (i_random + 1) * self.nb_q]
            qdot_this_time = x_single[self.q_offset + i_random * self.nb_q : self.q_offset + (i_random + 1) * self.nb_q]
            motor_noise_this_time = noise_single[noise_offset : noise_offset + self.nb_q]
            noise_offset += self.nb_q
            sensory_noise_this_time = noise_single[
                noise_offset : noise_offset + self.n_references
            ]
            noise_offset += self.n_references

            # Collect tau components
            torques_computed = self.collect_tau(
                q_this_time,
                qdot_this_time,
                muscle_activations,
                k_fb,
                ref_fb,
                tau,
                motor_noise_this_time,
                sensory_noise_this_time
            )

            # Dynamics
            qddot[i_random * self.nb_q : (i_random + 1) * self.nb_q] = self.forward_dynamics(
                q_this_time, qdot_this_time, torques_computed
            )

        dxdt = cas.vertcat(x_single[self.q_offset :], qddot)
        return dxdt
