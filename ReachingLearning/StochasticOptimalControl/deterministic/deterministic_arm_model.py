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
            sensory_noise_magnitude=np.zeros((4, 1)),
            motor_noise_magnitude=np.zeros((6, 1)),
            force_field_magnitude=force_field_magnitude,
        )

    def dynamics(
        self,
        x_single,
        u_single,
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

        q = x_single[0:2]
        qdot = x_single[2:4]
        muscle = u_single

        muscles_tau = self.get_muscle_torque(q, qdot, muscle)

        tau_force_field = self.force_field(q, self.force_field_magnitude)

        torques_computed = muscles_tau + tau_force_field - self.friction_coefficients @ qdot

        qddot = self.forward_dynamics(q, qdot, torques_computed)

        dxdt = cas.vertcat(qdot, qddot)

        return dxdt

    # def _compute_torques_from_noise_and_feedback_default(
    #     model, time, q, qdot, muscle, fb_ref, fb_gains, ff_ref, ff_gains, sensory_noise, motor_noise
    # ):
    #
    #     tau_from_muscles
    #
    #     ref = DynamicsFunctions.get(nlp.algebraic_states["ref"], algebraic_states)
    #     k = DynamicsFunctions.get(nlp.algebraic_states["k"], algebraic_states)
    #     k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)
    #
    #     sensory_input = model.sensory_reference(model, time, q, qdot, muscle, fb_ref, fb_gains, ff_ref, ff_gains, sensory_noise, motor_noise)
    #     tau_fb = k_matrix @ ((sensory_input - ref) + sensory_noise)
    #
    #     tau_motor_noise = motor_noise
    #
    #     tau = tau_nominal + tau_fb + tau_motor_noise
    #
    #     return tau
