import casadi as cas

from ..stochastic_basic.stochastic_basic_arm_model import StochasticBasicArmModel


class StochasticDelayArmModel(StochasticBasicArmModel):
    def __init__(
        self,
        motor_noise_std: float,
        wPq_std: float,
        wPqdot_std: float,
        dt: float,
        delay: float,
        force_field_magnitude: float = 0,
        n_random: int = 20,
    ):
        super().__init__(motor_noise_std, wPq_std, wPqdot_std, dt, force_field_magnitude, n_random)

        if delay % dt != 0:
            raise ValueError(
                f"The delay {delay} must be a multiple of the time step (final_time/n_shooting = {dt:.4f}s)."
            )
        self.delay = delay
        self.nb_frames_delay = int(delay / dt)

    def collect_tau(self, q, q_ee_delay, qdot, qdot_ee_delay, muscle_activations, k_fb, ref_fb, sensory_noise):
        """
        Collect all tau components
        """
        muscles_tau = self.get_muscle_torque(q, qdot, muscle_activations)
        tau_force_field = self.force_field(q, self.force_field_magnitude)
        tau_fb = k_fb @ (self.sensory_output(q_ee_delay, qdot_ee_delay, sensory_noise) - ref_fb)
        tau_friction = -self.friction_coefficients @ qdot
        torques_computed = muscles_tau + tau_force_field + tau_fb + tau_friction
        return torques_computed

    def dynamics(
        self,
        x_single,
        x_ee_delay_single,
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
        Noises:
        - motor_noise (6 x n_random, n_shooting)
        - sensory_noise (4 x n_random, n_shooting)
        """

        # Collect variables
        muscle = u_single[: self.nb_muscles]
        muscle_offset = self.nb_muscles
        k_fb = self.reshape_vector_to_matrix(
            u_single[muscle_offset : muscle_offset + self.n_references * self.nb_q], self.matrix_shape_k_fb
        )
        k_fb_offset = muscle_offset + self.n_references * self.nb_q
        ref_fb = u_single[k_fb_offset : k_fb_offset + self.n_references]
        qddot = cas.MX.zeros(self.nb_q * self.n_random)
        noise_offset = 0
        for i_random in range(self.n_random):
            q_this_time = x_single[i_random * self.nb_q : (i_random + 1) * self.nb_q]
            q_ee_delay_this_time = x_ee_delay_single[i_random * self.nb_q : (i_random + 1) * self.nb_q]
            qdot_this_time = x_single[self.q_offset + i_random * self.nb_q : self.q_offset + (i_random + 1) * self.nb_q]
            qdot_ee_delay_this_time = x_ee_delay_single[
                self.q_offset + i_random * self.nb_q : self.q_offset + (i_random + 1) * self.nb_q
            ]
            motor_noise_this_time = noise_single[noise_offset : noise_offset + self.nb_muscles]
            sensory_noise_this_time = noise_single[
                noise_offset + self.nb_muscles : noise_offset + self.nb_muscles + self.n_references
            ]
            noise_offset += self.n_noises

            # Get the real muscle activations (noised and avoid negative values)
            noised_muscle_activations = muscle + motor_noise_this_time
            noised_muscle_activations = cas.fabs(noised_muscle_activations)  # In [0, inf[ instead of [1e-6, 1]

            # Collect tau components
            torques_computed = self.collect_tau(
                q_this_time,
                q_ee_delay_this_time,
                qdot_this_time,
                qdot_ee_delay_this_time,
                noised_muscle_activations,
                k_fb,
                ref_fb,
                sensory_noise_this_time,
            )

            # Dynamics
            qddot[i_random * self.nb_q : (i_random + 1) * self.nb_q] = self.forward_dynamics(
                q_this_time, qdot_this_time, torques_computed
            )

        dxdt = cas.vertcat(x_single[self.q_offset :], qddot)
        return dxdt
