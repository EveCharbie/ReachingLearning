"""
This file contains the model used in the article.
"""

import casadi as cas
import numpy as np
import biorbd_casadi as biorbd


class ArmModel:
    """
    This allows to generate the same model as in the paper.
    """

    def __init__(
        self,
        force_field_magnitude: float = 0.0,
        n_random: int = 1,
    ):
        # Add the biorbd model here
        self.biorbd_model = biorbd.Model("../ReachingLearning/StochasticOptimalControl/models/arm_model.bioMod")

        self.n_random = n_random
        self.force_field_magnitude = force_field_magnitude

        # self.Faparam = np.array(
        #     [
        #         0.814483478343008,
        #         1.055033428970575,
        #         0.162384573599574,
        #         0.063303448465465,
        #         0.433004984392647,
        #         0.716775413397760,
        #         -0.029947116970696,
        #         0.200356847296188,
        #     ]
        # )
        # self.Fvparam = np.array([-0.318323436899127, -8.149156043475250, -0.374121508647863, 0.885644059915004])
        # self.Fpparam = np.array([-0.995172050006169, 53.598150033144236])
        # self.muscleDampingCoefficient = np.ones((6, 1)) * 0.01
        #
        #
        # # Active muscle force-length characteristic
        # self.b11 = self.Faparam[0]
        # self.b21 = self.Faparam[1]
        # self.b31 = self.Faparam[2]
        # self.b41 = self.Faparam[3]
        # self.b12 = self.Faparam[4]
        # self.b22 = self.Faparam[5]
        # self.b32 = self.Faparam[6]
        # self.b42 = self.Faparam[7]
        # self.b13 = 0.1
        # self.b23 = 1
        # self.b33 = 0.5 * cas.sqrt(0.5)
        # self.b43 = 0
        #
        # self.e0 = 0.6
        # self.e1 = self.Fvparam[0]
        # self.e2 = self.Fvparam[1]
        # self.e3 = self.Fvparam[2]
        # self.e4 = self.Fvparam[3]
        #
        # self.kpe = 4
        # self.tau_coef = 0.1500

        self.friction_coefficients = np.array([[0.05, 0.025], [0.025, 0.05]])

    @staticmethod
    def reshape_matrix_to_vector(matrix: cas.MX | cas.DM) -> cas.MX | cas.DM:
        matrix_shape = matrix.shape
        vector = type(matrix)()
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(matrix_shape[1]):
                vector = cas.vertcat(vector, matrix[i_shape, j_shape])
        return vector

    @staticmethod
    def reshape_vector_to_matrix(vector: cas.MX | cas.DM, matrix_shape: tuple[int, ...]) -> cas.MX | cas.DM:
        matrix = type(vector).zeros(matrix_shape)
        idx = 0
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(matrix_shape[1]):
                matrix[i_shape, j_shape] = vector[idx]
                idx += 1
        return matrix

    def get_mean_q(self, x):
        q = x[:self.nb_q]
        for i_random in range(1, self.n_random):
            q = cas.horzcat(q, x[i_random * self.nb_q:(i_random + 1) * self.nb_q])
        q_mean = cas.sum2(q) / self.n_random
        return q_mean

    def get_mean_qdot(self, x):
        qdot = x[self.q_offset : self.q_offset + self.nb_q]
        for i_random in range(1, self.n_random):
            qdot = cas.horzcat(qdot, x[self.q_offset + i_random * self.nb_q : self.q_offset + (i_random + 1) * self.nb_q])
        qdot_mean = cas.sum2(qdot) / self.n_random
        return qdot_mean

    @property
    def nb_muscles(self):
        return 6

    @property
    def nb_q(self):
        return 2

    @property
    def nb_qdot(self):
        return 2

    @property
    def name_dof(self):
        return ["shoulder", "elbow"]

    @property
    def muscle_names(self):
        return [f"muscle_{i}" for i in range(self.nb_muscles)]

    @property
    def q_offset(self):
        return self.nb_q * self.n_random

    # def get_muscle_length(self, q):
    #     """
    #     Get the muscle lengths given the joint angles
    #     """
    #     muscle_lengths = []
    #     for muscle in self.biorbd_model.muscles:
    #         muscle_lengths += muscle.length(q)
    #     return muscle_lengths

    def get_muscle_torque(self, q: cas.MX, qdot: cas.MX, mus_activations: cas.MX) -> cas.MX:
        muscles_states = self.biorbd_model.stateSet()
        for k in range(self.biorbd_model.nbMuscles()):
            muscles_states[k].setActivation(mus_activations[k])
        q_biorbd = biorbd.GeneralizedCoordinates(q)
        qdot_biorbd = biorbd.GeneralizedVelocity(qdot)
        return self.biorbd_model.muscularJointTorque(muscles_states, q_biorbd, qdot_biorbd).to_mx()

    def forward_dynamics(self, q: cas.MX, qdot: cas.MX, tau: cas.MX) -> cas.MX:
        return self.biorbd_model.ForwardDynamics(q, qdot, tau).to_mx()

    def force_field(self, q, force_field_magnitude):
        """
        Since the model is centered on the shoulder, the force field is simply proportional to the hand position.
        """
        return force_field_magnitude * self.end_effector_position(q)

    def get_excitation_feedback(self, K, EE, EE_ref, sensory_noise):
        return K @ ((EE - EE_ref) + sensory_noise)

    def end_effector_position(self, q: cas.MX) -> cas.MX:
        marker_index = biorbd.marker_index(self.biorbd_model, "end_effector")
        q_biorbd = biorbd.GeneralizedCoordinates(q)
        return self.biorbd_model.marker(q_biorbd, marker_index).to_mx()[:2]

    def end_effector_velocity(self, q: cas.MX, qdot: cas.MX) -> cas.MX:
        marker_index = biorbd.marker_index(self.biorbd_model, "end_effector")
        q_biorbd = biorbd.GeneralizedCoordinates(q)
        qdot_biorbd = biorbd.GeneralizedVelocity(qdot)
        return self.biorbd_model.markerVelocity(q_biorbd, qdot_biorbd, marker_index).to_mx()[:2]

    def end_effector_pos_velo(self, q, qdot) -> cas.MX:
        hand_pos = self.end_effector_position(q)
        hand_vel = self.end_effector_velocity(q, qdot)
        ee = cas.vertcat(hand_pos, hand_vel)
        return ee
