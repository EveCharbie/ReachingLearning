import pickle
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import casadi as cas


from .utils import (
    get_the_real_dynamics,
    integrate_the_dynamics,
    generate_random_data,
    get_the_real_marker_position,
    sample_task_from_circle,
    integrate_MS,
    animate_reintegration,
)
# from ...constants import TARGET_START_VAN_WOUWE, TARGET_END_VAN_WOUWE
from .spline_dynamics_parameters import get_tau_opt

from ...StochasticOptimalControl.deterministic.deterministic_OCP import prepare_ocp
from ...StochasticOptimalControl.deterministic.deterministic_save_results import save_ocp_torque_driven
from ...StochasticOptimalControl.deterministic.deterministic_confirm_solution import confirm_optimal_solution_ocp
from ...StochasticOptimalControl.utils import ExampleType, solve, get_dm_value


class SplineParametersDynamicsLearner:
    """
    Spline approximation of the dynamics.
    """

    def __init__(
            self,
            nb_q: int,
            smoothness: float = 0.1,
            nb_grid_points_q: int = 10,
            nb_grid_points_qdot: int = 20,
            max_velocity: float = 10 * np.pi,
            kernel: str = 'thin_plate_spline',
    ):
        self.nb_q = nb_q
        self.smoothness = smoothness
        self.nb_grid_points_q = nb_grid_points_q
        self.nb_grid_points_qdot = nb_grid_points_qdot
        self.max_velocity = max_velocity
        self.kernel = kernel

        # Storage for training data
        self.input_training_data = None
        self.output_training_data = {"M11": None, "M12": None, "M22": None, "N1": None, "N2": None}

        # Spline estimates
        self.spline_model = {"M11": None, "M12": None, "M22": None, "N1": None, "N2": None}
        self.xdot_errors = []
        self.reintegration_errors = []

    def from_learned_parameters(self, learned_parameters: dict[str, any]) -> "Self":
        """Load learned parameters from a dictionary"""
        self.input_training_data = learned_parameters["input_training_data"]
        self.output_training_data = learned_parameters["output_training_data"]
        self.spline_model = learned_parameters["spline_model"]
        self.xdot_errors = learned_parameters["xdot_errors"]
        self.reintegration_errors = learned_parameters["reintegration_errors"]
        return self

    def update(self, x_samples, M_real, N_real):
        """
        Update the spline models with new observations.

        Parameters
        ----------
        x_samples: array of shape (n_samples, nb_q * 2) - states
        M_real: array of shape (n_samples, 2, 2) - true mass matrix
        N_real: array of shape (n_samples, 2) - true non-linear effects vector
        """
        # Get the good samples
        good_indices = np.where(np.logical_and(np.logical_and(np.logical_and(
            x_samples[:, 0] > 0,
            x_samples[:, 0] < np.pi / 2),
            x_samples[:, 1] > 0),
            x_samples[:, 1] < 7/8 * np.pi)
        )
        if len(good_indices[0]) == 0:
            return
        else:
            input_new = x_samples[good_indices[0], :]

            # Add to training data
            if self.input_training_data is None:
                self.input_training_data = input_new
            else:
                self.input_training_data = np.vstack((self.input_training_data, input_new))

            if self.output_training_data["M11"] is None:
                self.output_training_data["M11"] = M_real[good_indices[0], 0, 0]
                self.output_training_data["M12"] = M_real[good_indices[0], 0, 1]
                self.output_training_data["M22"] = M_real[good_indices[0], 1, 1]
                self.output_training_data["N1"] = N_real[good_indices[0], 0]
                self.output_training_data["N2"] = N_real[good_indices[0], 1]
            else:
                self.output_training_data["M11"] = np.hstack((self.output_training_data["M11"], M_real[good_indices[0], 0, 0]))
                self.output_training_data["M12"] = np.hstack((self.output_training_data["M12"], M_real[good_indices[0], 0, 1]))
                self.output_training_data["M22"] = np.hstack((self.output_training_data["M22"], M_real[good_indices[0], 1, 1]))
                self.output_training_data["N1"] = np.hstack((self.output_training_data["N1"], N_real[good_indices[0], 0]))
                self.output_training_data["N2"] = np.hstack((self.output_training_data["N2"], N_real[good_indices[0], 1]))

            # Retrain the spline model
            for key in ["M11", "M12", "M22", "N1", "N2"]:
                worked = False
                while not worked and self.input_training_data.shape[0] > 4:
                    try:
                        self.spline_model[key] = RBFInterpolator(
                            self.input_training_data,
                            self.output_training_data[key],
                            smoothing=self.smoothness,
                            kernel=self.kernel,
                        )
                        worked = True
                    except:
                        print(f"Warning: could not fit spline for {key} with current data, so we remove data points.")
                        worked = False
                        self.input_training_data = self.input_training_data[:-1, :]
                        self.output_training_data[key] = self.output_training_data[key][:-1]

    def add_errors(self, xdot_errors, reintegration_errors):
        """Add an error measurement"""
        self.xdot_errors.append(xdot_errors)
        self.reintegration_errors.append(reintegration_errors)

    def predict(self, x, u):
        """
        Predict state derivative xdot = f(x, u) using the learned spline parameters model.
        This function is to be used to evaluate the dynamics after the optimization

        Parameters
        ----------
        x: state vector of shape (2 * nb_q,)
        u: control vector of shape (nb_q,)

        Returns
        -------
        xdot_estimate: predicted state derivative
        """
        # Create feature vector
        input_test = x.reshape(1, -1)

        M11 = np.array(self.spline_model["M11"](input_test)[0])
        M12 = np.array(self.spline_model["M12"](input_test)[0])
        M22 = np.array(self.spline_model["M22"](input_test)[0])
        inv_mass_matrix = np.array([[M11, M12],
                                [M12, M22]]).reshape(2, 2)

        N1 = np.array(self.spline_model["N1"](input_test)[0])
        N2 = np.array(self.spline_model["N2"](input_test)[0])
        nonlinear_effects = np.array([N1, N2]).reshape(2, )

        dq = x[self.nb_q:].reshape(2, 1)
        dqdot = inv_mass_matrix @ np.reshape(u - nonlinear_effects, (2, 1))
        xdot_estimate = np.vstack((dq, dqdot))
        return xdot_estimate.reshape(4, )

    def forward_dyn(self, x, u, motor_noise):
        return self.predict(x, u)

    def casadi_forward_dyn_func(self, iter_num: int) -> cas.Function:
        """
        Create an interpolant in casadi by evaluating the scipy one on an equal grid.
        This function is to be used in the OCP.
        """

        # Creating a fixed meshgrid over the RoM as CasADi is a fucker and returns zero instead of extrapolating !
        input_resampled_q1 = np.linspace(0, np.pi/2, self.nb_grid_points_q)
        input_resampled_q2 = np.linspace(0, 7/8 * np.pi, self.nb_grid_points_q)
        input_resampled_qdot1 = np.linspace(-self.max_velocity, self.max_velocity, self.nb_grid_points_qdot)
        input_resampled_qdot2 = np.linspace(-self.max_velocity, self.max_velocity, self.nb_grid_points_qdot)
        gridq1, gridq2, gridqdot1, gridqdot2 = np.meshgrid(
            input_resampled_q1,
            input_resampled_q2,
            input_resampled_qdot1,
            input_resampled_qdot2,
            indexing='ij',
        )

        # Get the interpolated values
        interpolated_values = {}
        lut = {}
        for key in ["M11", "M12", "M22", "N1", "N2"]:
            interpolated_values[key] = self.spline_model[key](np.vstack((
                gridq1.ravel(),
                gridq2.ravel(),
                gridqdot1.ravel(),
                gridqdot2.ravel(),
            )).T)
            lut[key] = cas.interpolant(f"lut_{iter_num}_{key}", "bspline", [
                    input_resampled_q1,
                    input_resampled_q2,
                    input_resampled_qdot1,
                    input_resampled_qdot2,
                ], interpolated_values[key])

        X = cas.MX.sym("X", self.nb_q * 2)
        # U = cas.MX.sym("U", 6)
        U = cas.MX.sym("U", 2)

        # # TODO: remove this portion and include muscle dynamics in the learned model
        # import biorbd_casadi
        # current_path = Path(__file__).parent
        # model_path = f"{current_path}/../../StochasticOptimalControl/models/arm_model.bioMod"
        # biorbd_model = biorbd_casadi.Biorbd(model_path)
        # tau = biorbd_model.muscles.joint_torque(activations=U, q=X[:self.nb_q], qdot=X[self.nb_q:])

        M11 = lut["M11"](X)
        M12 = lut["M12"](X)
        M22 = lut["M22"](X)
        inv_mass_matrix = cas.vertcat(
            cas.horzcat(M11, M12),
            cas.horzcat(M12, M22)
        )

        N1 = lut["N1"](X)
        N2 = lut["N2"](X)
        nonlinear_effects = cas.vertcat(N1, N2)

        dq = X[self.nb_q:].reshape((2, 1))
        dqdot = inv_mass_matrix @ cas.reshape(U - nonlinear_effects, 2, 1)
        xdot_estimate = cas.vertcat(dq, dqdot)
        forward_dyn_fcn = cas.Function("forward_dyn", [X, U], [xdot_estimate])
        return forward_dyn_fcn

    def save_model(self, str_sup: str = ""):
        """
        Save the learned model to a file.
        """
        current_path = Path(__file__).parent
        spline_model_path = f"{current_path}/../../../results/LearningInternalDynamics/spline_dynamics_model_{str_sup}.pkl"

        with open(spline_model_path, 'wb') as f:
            output = {
                "input_training_data": self.input_training_data,
                "output_training_data": self.output_training_data,
                "spline_model": self.spline_model,
                "xdot_errors": self.xdot_errors,
                "reintegration_errors": self.reintegration_errors,
            }
            pickle.dump(output, f)


def train_spline_dynamics_parameters_ocp(smoothness: float, nb_grid_points_q: int, nb_grid_points_qdot: int):
    """
    Train a spline dynamics parameters model using OCPs to generate the xdots.

    Parameters
    ----------
    smoothness: float
        Smoothing factor for the spline interpolation. Ranging between 0 (not smoothened), and 1 (very filtered like).
    nb_grid_points: int
        Number of grid points to use for the spline reinterpolation when going from scipy to casadi LUT.
    """
    # TODO: add uncertainty (modeling) on the end_effector_position function
    np.random.seed(42)

    # Constants
    final_time = 0.8
    dt = 0.05
    nb_q = 2
    n_shooting = int(final_time / dt)
    hand_error_threshold = 0.01  # 1 cm
    tol = 1e-6
    # motor_noise = 0.05
    max_tau = 10.0
    max_velocity = 5.0

    # Get the real dynamics
    real_forward_dyn, inv_mass_matrix_func, nl_effect_vector_func = get_the_real_dynamics()
    real_marker_func = get_the_real_marker_position()

    # Initialize the Spline parameter learner
    learner = SplineParametersDynamicsLearner(
        nb_q=nb_q,
        smoothness=smoothness,
        nb_grid_points_q=nb_grid_points_q,
        nb_grid_points_qdot=nb_grid_points_qdot,
        max_velocity=max_velocity,
    )

    # Learn ten episodes
    for i_learn in range(10):

        # Generate random data to initially train on
        x0_this_time, u_this_time = generate_random_data(nb_q, n_shooting, max_tau=max_tau, max_velocity=max_velocity)

        # Evaluate the error made by the approximate dynamics
        _, x_integrated_real, _, xdot_real, M_real, N_real = integrate_the_dynamics(
            x0_this_time,
            u_this_time,
            dt,
            current_forward_dyn=None,
            real_forward_dyn=real_forward_dyn,
            inv_mass_matrix_func=inv_mass_matrix_func,
            nl_effect_vector_func=nl_effect_vector_func,
        )

        # Update the Spline parameter model with new observations
        learner.update(
            x_samples=x_integrated_real[:, :-1].T,
            M_real=M_real,
            N_real=N_real,
        )

    # Set up the output file and redirect printing to this file
    current_path = Path(__file__).parent
    output_file_path = f"{current_path}/../../../results/LearningInternalDynamics/spline_dynamics_parameters_OCP.txt"
    output_file = open(output_file_path, 'w')
    sys.stdout = output_file

    # Track learning progress
    hand_position_error = np.inf
    i_episode = 0
    while hand_position_error > hand_error_threshold and i_episode < 250:

        # Sample a task
        target_start, target_end = sample_task_from_circle()
        # target_start = TARGET_START_VAN_WOUWE
        # target_end = TARGET_END_VAN_WOUWE

        # Generate reaching movement
        forward_dynamics_func = learner.casadi_forward_dyn_func(i_episode)
        ocp = prepare_ocp(
            final_time=final_time,
            n_shooting=n_shooting,
            example_type=ExampleType.CIRCLE,
            forward_dynamics_func=forward_dynamics_func,
            target_start=target_start,
            target_end=target_end,
            muscle_driven=False,
            max_velocity=max_velocity,  # Reduced max velocity to help calm the dynamics
        )
        w_opt, f_opt, g_opt, solver = solve(
            ocp,
            tol=tol,
            pre_optim_plot=False,
            show_online_optim=False,
            max_iter=100,
        )
        if solver.stats()["success"]:
            # We use the optimal solution
            save_path_ocp = f"{current_path}/../../../results/LearningInternalDynamics/ocp_results_spline_dynamics_parameters_{i_episode}.pkl"
            variable_data = save_ocp_torque_driven(w_opt, ocp, save_path_ocp, tol, solver)
            # confirm_optimal_solution_ocp(w_opt, f_opt, g_opt, ocp)

            x0_this_time = np.hstack((variable_data["q_opt"][:, 0], variable_data["qdot_opt"][:, 0]))
            u_this_time = variable_data["tau_opt"]
            time_vector = variable_data["time_vector"]

            # Get the hand trajectory of the optimal solution
            opt_end_effector_position = np.zeros((2, n_shooting + 1))
            for i_shooting in range(n_shooting + 1):
                opt_end_effector_position[:, i_shooting] = np.array(real_marker_func(variable_data["q_opt"][:, i_shooting])).reshape(2, )
            converged = True
        else:
            # We generate random data to train the model since the OCP did not converge
            x0_this_time, u_this_time = generate_random_data(nb_q, n_shooting)
            time_vector = np.linspace(0, final_time, n_shooting + 1)
            opt_end_effector_position = np.zeros((2, n_shooting + 1))
            converged = False

        # Evaluate the error made by the approximate dynamics
        # current_forward_dyn = learner.forward_dyn
        x_sym = cas.MX.sym("x", nb_q * 2)
        u_sym = cas.MX.sym("u", 2)
        motor_noise_sym = cas.MX.sym("motor_noise", 2)
        current_forward_dyn = cas.Function("forward_dyn", [x_sym, u_sym, motor_noise_sym], [ocp["model"].dynamics(x_sym, u_sym)])
        x_integrated_approx, x_integrated_real, xdot_approx, xdot_real, M_real, N_real = integrate_the_dynamics(
            x0_this_time,
            u_this_time,
            dt,
            current_forward_dyn=current_forward_dyn,
            real_forward_dyn=real_forward_dyn,
            inv_mass_matrix_func=inv_mass_matrix_func,
            nl_effect_vector_func=nl_effect_vector_func,
        )
        x_integrated_approx_MS, x_integrated_real_MS = integrate_MS(
            x_opt=np.vstack((variable_data["q_opt"], variable_data["qdot_opt"])),
            u=u_this_time,
            dt=dt,
            current_forward_dyn=current_forward_dyn,
            real_forward_dyn=real_forward_dyn,
        )
        time_vector_MS = np.zeros((6 * n_shooting, ))
        for i_node in range(n_shooting):
            time_vector_MS[6 * i_node :6 * (i_node + 1)] = np.linspace(time_vector[i_node], time_vector[i_node + 1], 6)

        # Compute the dynamics error
        xdot_error_norm = np.linalg.norm(xdot_approx - xdot_real, axis=0) * 180 / np.pi
        xdot_errors_this_time = np.mean(xdot_error_norm)
        reintegration_error_norm = np.linalg.norm(x_integrated_approx - x_integrated_real, axis=0) * 180 / np.pi
        reintegration_errors_this_time = np.mean(reintegration_error_norm)
        learner.add_errors(xdot_errors_this_time, reintegration_errors_this_time)

        # Compute the hand position error at final time
        hand_position_real = np.zeros((2, n_shooting + 1))
        for i_shooting in range(n_shooting + 1):
            hand_position_real[:, i_shooting] = np.array(
                real_marker_func(x_integrated_real[:2, i_shooting])).reshape(2, )
        hand_position_error = np.linalg.norm(target_end - hand_position_real[:, -1])

        # # Animate the optimal solution reintegration
        # animate_reintegration(
        #     q_reintegrated=x_integrated_real[:2, :],
        #     muscles_opt=u_this_time,
        # )

        # Plot the results
        fig, axs = plt.subplots(2, 3, figsize=(10, 8))
        axs[0, 0].plot(time_vector, x_integrated_approx[0, :], '-c', label='Approx x')
        axs[0, 0].plot(time_vector, x_integrated_real[0, :], '--b', label='Real x')
        axs[0, 0].plot(time_vector[:-1], xdot_approx[0, :], '-m', label='Approx xdot')
        axs[0, 0].plot(time_vector[:-1], xdot_real[0, :], '--r', label='Real xdot')
        axs[0, 0].plot(time_vector_MS, x_integrated_approx_MS[0, :], '.', color="tab:gray", label='Approx x MS')
        # axs[0, 0].plot(time_vector_MS, x_integrated_real_MS[0, :], 'o', mfc='none', color="k", label='real x MS')
        if converged:
            axs[0, 0].plot(time_vector, variable_data["q_opt"][0, :], ':g', label='Optimal xdot')
        axs[0, 0].set_title("Q1")

        axs[0, 1].plot(time_vector, x_integrated_approx[1, :], '-c', label='Approx x')
        axs[0, 1].plot(time_vector, x_integrated_real[1, :], '--b', label='Real x')
        axs[0, 1].plot(time_vector[:-1], xdot_approx[1, :], '-m', label='Approx xdot')
        axs[0, 1].plot(time_vector[:-1], xdot_real[1, :], '--r', label='Real xdot')
        axs[0, 1].plot(time_vector_MS, x_integrated_approx_MS[1, :], '.', color="tab:gray", label='Approx x MS')
        # axs[0, 1].plot(time_vector_MS, x_integrated_real_MS[1, :], 'o', mfc='none', color="k", label='real x MS')
        if converged:
            axs[0, 1].plot(time_vector, variable_data["q_opt"][1, :], ':g', label='Optimal x')
        axs[0, 1].set_title("Q2")

        axs[1, 0].plot(time_vector, x_integrated_approx[2, :], '-c', label='Approx x')
        axs[1, 0].plot(time_vector, x_integrated_real[2, :], '--b', label='Real x')
        axs[1, 0].plot(time_vector[:-1], xdot_approx[2, :], '-m', label='Approx xdot')
        axs[1, 0].plot(time_vector[:-1], xdot_real[2, :], '--r', label='Real xdot')
        axs[1, 0].plot(time_vector_MS, x_integrated_approx_MS[2, :], '.', color="tab:gray", label='Approx x MS')
        # axs[1, 0].plot(time_vector_MS, x_integrated_real_MS[2, :], 'o', mfc='none', color="k", label='real x MS')
        if converged:
            axs[1, 0].plot(time_vector, variable_data["qdot_opt"][0, :], ':g', label='Optimal x')
        axs[1, 0].set_title("Qdot1")

        axs[1, 1].plot(time_vector, x_integrated_approx[3, :], '-c', label='Approx x')
        axs[1, 1].plot(time_vector, x_integrated_real[3, :], '--b', label='Real x')
        axs[1, 1].plot(time_vector[:-1], xdot_approx[3, :], '-m', label='Approx xdot')
        axs[1, 1].plot(time_vector[:-1], xdot_real[3, :], '--r', label='Real xdot')
        axs[1, 1].plot(time_vector_MS, x_integrated_approx_MS[3, :], '.', color="tab:gray", label='Approx x MS')
        # axs[1, 1].plot(time_vector_MS, x_integrated_real_MS[3, :], 'o', mfc='none', color="k", label='real x MS')
        if converged:
            axs[1, 1].plot(time_vector, variable_data["qdot_opt"][1, :], ':g', label='Optimal x')
        axs[1, 1].set_title("Qdot2")

        axs[0, 1].legend()

        axs[0, 2].plot(hand_position_real[0, :], hand_position_real[1, :], '--b', label='Real hand trajectory')
        axs[0, 2].plot(opt_end_effector_position[0, :], opt_end_effector_position[1, :], ':g', label='Optimal hand trajectory')
        axs[0, 2].plot(target_start[0], target_start[1], 'og', label='Target')
        axs[0, 2].plot(target_end[0], target_end[1], 'or', label='Target')
        home_position = np.array([-0.0212132, 0.445477])
        circ = plt.Circle(home_position, 0.15, fill=False, linestyle="-")
        axs[0, 2].add_patch(circ)
        axs[0, 2].axis("equal")

        # Creating a fixed meshgrid over the RoM as CasADi is a fucker and returns zero instead of extrapolating !
        nb_resampling = 30
        input_resampled = np.zeros((nb_resampling, 2 * 2))
        input_resampled[:, 0] = np.linspace(0, np.pi/2, nb_resampling)
        input_resampled[:, 1] = np.linspace(0, 7/8 * np.pi, nb_resampling)
        input_resampled[:, 2] = np.linspace(-2 * np.pi, 2 * np.pi, nb_resampling)
        input_resampled[:, 3] = np.linspace(-2 * np.pi, 2 * np.pi, nb_resampling)
        gridq1, gridq2 = np.meshgrid(
            input_resampled[:, 0],
            input_resampled[:, 1],
            indexing='ij',
        )
        gridqdot1, gridqdot2 = np.meshgrid(
            input_resampled[:, 2],
            input_resampled[:, 3],
            indexing='ij',
        )

        # Get the interpolated values
        interpolated_values_M11 = learner.spline_model["M11"](np.vstack((
                gridq1.ravel(),
                gridq2.ravel(),
                np.ones_like(gridq1.ravel()),
                np.ones_like(gridq2.ravel()),
            )).T)
        interpolated_values_M12 = learner.spline_model["M12"](np.vstack((
                gridq1.ravel(),
                gridq2.ravel(),
                np.ones_like(gridq1.ravel()),
                np.ones_like(gridq2.ravel()),
            )).T)
        interpolated_values_M22 = learner.spline_model["M22"](np.vstack((
                gridq1.ravel(),
                gridq2.ravel(),
                np.ones_like(gridq1.ravel()),
                np.ones_like(gridq2.ravel()),
            )).T)
        interpolated_values_N2 = learner.spline_model["N2"](np.vstack((
                gridq1.ravel(),
                gridq2.ravel(),
                np.ones_like(gridq1.ravel()),
                np.ones_like(gridq2.ravel()),
            )).T)
        interpolated_values_N1 = learner.spline_model["N1"](np.vstack((
                gridq1.ravel(),
                gridq2.ravel(),
                np.ones_like(gridq1.ravel()),
                np.ones_like(gridq2.ravel()),
            )).T)
        interpolated_values_N2V = learner.spline_model["N2"](np.vstack((
                np.ones_like(gridq1.ravel()),
                np.ones_like(gridq2.ravel()),
                gridqdot1.ravel(),
                gridqdot2.ravel(),
            )).T)
        interpolated_values_N1V = learner.spline_model["N1"](np.vstack((
                np.ones_like(gridq1.ravel()),
                np.ones_like(gridq2.ravel()),
                gridqdot1.ravel(),
                gridqdot2.ravel(),
            )).T)

        axs[1, 2].remove()
        ax = fig.add_subplot(2, 3, 6, projection='3d')
        # ax.plot_surface(gridq1[:, :, 0, 0], gridq2[:, :, 0, 0], interpolated_values_M12.reshape(30, 30, 30, 30)[:, :, 0, 0])
        # ax.plot_surface(gridq1[:, :, 0, 0], gridq2[:, :, 0, 0], interpolated_values_N2.reshape(30, 30, 30, 30)[:, :, 0, 0])
        # ax.plot_surface(gridqdot1[0, 0, :, :], gridqdot2[0, 0, :, :], interpolated_values_N2.reshape(30, 30, 30, 30)[0, 0, :, :])
        ax.plot(gridq1.ravel(), gridq2.ravel(), interpolated_values_M11, '.r', markersize=0.5)
        ax.plot(gridq1.ravel(), gridq2.ravel(), interpolated_values_M12, '.m', markersize=0.5)
        ax.plot(gridq1.ravel(), gridq2.ravel(), interpolated_values_M22, '.', color="tab:orange", markersize=0.5)
        ax.plot(gridq1.ravel(), gridq2.ravel(), interpolated_values_N1, '.b', markersize=0.5)
        ax.plot(gridq1.ravel(), gridq2.ravel(), interpolated_values_N2, '.c', markersize=0.5)
        ax.plot(gridqdot1.ravel(), gridqdot2.ravel(), interpolated_values_N1V, '.b', markersize=0.5)
        ax.plot(gridqdot1.ravel(), gridqdot2.ravel(), interpolated_values_N2V, '.c', markersize=0.5)
        # ax.plot(gridqdot1.ravel(), gridqdot2.ravel(), interpolated_values_N2)

        fig_path = f"{current_path}/../../../results/LearningInternalDynamics/ocp_results_spline_dynamics_parameters_{i_episode}.png"
        plt.savefig(fig_path)
        # plt.show()
        plt.close()

        sys.stdout = sys.__stdout__
        print(f"{i_episode} --- reintegration error: "
              f"{reintegration_errors_this_time:.6f} [{np.min(reintegration_error_norm)}, {np.max(reintegration_error_norm)}] deg"
              f"--- hand position error: {hand_position_error * 100:.6f} cm")
        sys.stdout = output_file
        print(f"{i_episode} --- reintegration error: "
              f"{reintegration_errors_this_time:.6f} [{np.min(reintegration_error_norm)}, {np.max(reintegration_error_norm)}] deg"
              f"--- hand position error: {hand_position_error * 100:.6f} cm")

        # Update the Spline parameter model with new observations
        learner.update(
            x_samples=x_integrated_real[:, :-1].T,
            M_real=M_real,
            N_real=N_real,
        )
        i_episode += 1
        del ocp

    print("-----------------------------------------------------")
    print("Learning complete!")
    learner.save_model(str_sup=f"_OCP")
    # learner.plotter.stop()

    # Close the file and restore printing to the console
    sys.stdout = sys.__stdout__
    output_file.close()


def evaluate_spline_dynamics_parameters_ocp(smoothness: float, nb_grid_points_q: int, nb_grid_points_qdot: int):
    """Test loading the saved model"""

    # Load the model
    current_path = Path(__file__).parent
    spline_model_path = f"{current_path}/../../../results/LearningInternalDynamics/spline_dynamics_model__OCP.pkl"
    with open(spline_model_path, 'rb') as f:
        learned_parameters = pickle.load(f)
    learner = SplineParametersDynamicsLearner(
        nb_q=2,
        smoothness=smoothness,
        nb_grid_points_q=nb_grid_points_q,
        nb_grid_points_qdot=nb_grid_points_qdot,
        max_velocity=5,
        ).from_learned_parameters(learned_parameters)

    # Load the OCP kinematics to test the error
    ocp_file_path = f"{current_path}/../../../results/StochasticOptimalControl/ocp_forcefield0_CIRCLE_CVG_1p0e-06.pkl"
    with open(ocp_file_path, 'rb') as file:
        ocp_sol = pickle.load(file)

     # Extract the optimal trajectories
    q = ocp_sol["q_opt"]
    qdot = ocp_sol["qdot_opt"]
    muscles = ocp_sol["muscle_opt"]
    time_vector = ocp_sol["time_vector"]
    dt = time_vector[1] - time_vector[0]

    # Get the real dynamics
    real_forward_dyn, inv_mass_matrix_func, nl_effect_vector_func = get_the_real_dynamics()

    # TODO: this step should be added in the model (q, qdot, mus -> Tau)
    u = get_tau_opt(q=q, qdot=qdot, muscles=muscles)

    # Evaluate the error made by the approximate dynamics
    current_forward_dyn = learner.forward_dyn
    x_integrated_approx, x_integrated_real, xdot_approx, xdot_real, M_real, N_real = integrate_the_dynamics(
        np.hstack((q[:, 0], qdot[:, 0])),
        u,
        dt,
        current_forward_dyn=current_forward_dyn,
        real_forward_dyn=real_forward_dyn,
        inv_mass_matrix_func=inv_mass_matrix_func,
        nl_effect_vector_func=nl_effect_vector_func,
    )

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(time_vector, x_integrated_approx[0, :], '-c', label='Approx x')
    axs[0, 0].plot(time_vector, x_integrated_real[0, :], '-b', label='Real x')
    axs[0, 0].plot(time_vector[:-1], xdot_approx[0, :], '-m', label='Approx xdot')
    axs[0, 0].plot(time_vector[:-1], xdot_real[0, :], '-r', label='Real xdot')
    axs[0, 0].set_title("Q1")

    axs[0, 1].plot(time_vector, x_integrated_approx[1, :], '-c', label='Approx q')
    axs[0, 1].plot(time_vector, x_integrated_real[1, :], '-b', label='Real q')
    axs[0, 1].plot(time_vector[:-1], xdot_approx[1, :], '-m', label='Approx qdot')
    axs[0, 1].plot(time_vector[:-1], xdot_real[1, :], '-r', label='Real qdot')
    axs[0, 1].set_title("Q2")

    axs[1, 0].plot(time_vector, x_integrated_approx[2, :], '-c', label='Approx x')
    axs[1, 0].plot(time_vector, x_integrated_real[2, :], '-b', label='Real x')
    axs[1, 0].plot(time_vector[:-1], xdot_approx[2, :], '-m', label='Approx xdot')
    axs[1, 0].plot(time_vector[:-1], xdot_real[2, :], '-r', label='Real xdot')
    axs[1, 0].set_title("Qdot1")

    axs[1, 1].plot(time_vector, x_integrated_approx[3, :], '-c', label='Approx x')
    axs[1, 1].plot(time_vector, x_integrated_real[3, :], '-b', label='Real x')
    axs[1, 1].plot(time_vector[:-1], xdot_approx[3, :], '-m', label='Approx xdot')
    axs[1, 1].plot(time_vector[:-1], xdot_real[3, :], '-r', label='Real xdot')
    axs[1, 1].set_title("Qdot2")

    axs[0, 1].legend()
    fig_path = f"{current_path}/../../../figures/LearningInternalDynamics/spline_dynamics_model_evaluation__OCP.png"
    plt.savefig(fig_path)


