import pickle
from pathlib import Path
import threading

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

from .utils import get_the_real_dynamics, integrate_the_dynamics, generate_random_data


class LivePlotter:
    """Separate class to handle live plotting in a thread"""

    def __init__(self):
        self.fig = None
        self.axs = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

        # Data to plot
        self.input_training_data = []
        self.output_training_data = {"M11": None, "M12": None, "M22": None, "N1": None, "N2": None}
        self.spline_model = {"M11": None, "M12": None, "M22": None, "N1": None, "N2": None}
        self.xdot_errors = []
        self.reintegration_errors = []

    def start(self):
        """Start the plotting thread"""
        self.running = True
        self.thread = threading.Thread(target=self._plot_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the plotting thread"""
        self.running = False
        if self.thread:
            self.thread.join()

    def update_data(self, input_data=None, output_data=None, spline_model=None, reintegration_errors=None, xdot_errors=None):
        """Thread-safe data update"""
        with self.lock:
            if input_data is not None:
                self.input_training_data = input_data
            if output_data is not None:
                self.output_training_data = output_data
            if spline_model is not None:
                self.spline_model = spline_model
            if reintegration_errors is not None:
                self.reintegration_errors.append(reintegration_errors)
            if xdot_errors is not None:
                self.xdot_errors.append(xdot_errors)

    def _initialize_plots(self):
        """Initialize the plot window"""
        self.fig = plt.figure(figsize=(12, 8))
        ax00 = self.fig.add_subplot(2, 3, 1, projection='3d')
        ax10 = self.fig.add_subplot(2, 3, 2)
        ax20 = self.fig.add_subplot(2, 3, 3, projection='3d')
        ax01 = self.fig.add_subplot(2, 3, 4, projection='3d')
        ax11 = self.fig.add_subplot(2, 3, 5, projection='3d')
        ax21 = self.fig.add_subplot(2, 3, 6, projection='3d')
        self.axs = np.array([[ax00, ax10, ax20],
                             [ax01, ax11, ax21]])
        plt.ion()


    def _plot_loop(self):
        """Main plotting loop"""
        self._initialize_plots()

        while self.running:
            with self.lock:
                self._update_plots()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.1)

        plt.ioff()

    def _update_plots(self):
        """Update all plots with current data"""
        # Clear all axes
        for ax in self.axs.flat:
            ax.cla()

        output_pred_M11 = self.spline_model["M11"](self.input_training_data)
        output_pred_M12 = self.spline_model["M12"](self.input_training_data)
        output_pred_M22 = self.spline_model["M22"](self.input_training_data)
        output_pred_N1 = self.spline_model["N1"](self.input_training_data)
        output_pred_N2 = self.spline_model["N2"](self.input_training_data)

        if self.input_training_data is not None and self.output_training_data["M11"] is not None and self.spline_model["M11"] is not None:
            if self.input_training_data[:, 0].shape[0] == self.output_training_data["M11"].shape[0]:
                self.axs[0, 0].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], self.output_training_data["M11"], '.k', markersize=1)
                self.axs[0, 0].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], output_pred_M11, '.m', markersize=1)
                self.axs[0, 0].set_title("M11")
                self.axs[0, 0].set_xlabel("q1")
                self.axs[0, 0].set_ylabel("q2")

            if self.input_training_data[:, 0].shape[0] == self.output_training_data["M12"].shape[0]:
                self.axs[1, 0].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], self.output_training_data["M12"], '.k', markersize=1)
                self.axs[1, 0].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], output_pred_M12, '.m', markersize=1)
                self.axs[1, 0].set_title("M12")
                self.axs[1, 0].set_xlabel("q1")
                self.axs[1, 0].set_ylabel("q2")

            if self.input_training_data[:, 0].shape[0] == self.output_training_data["M22"].shape[0]:
                self.axs[1, 1].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], self.output_training_data["M22"], '.k', markersize=1)
                self.axs[1, 1].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], output_pred_M22, '.m', markersize=1)
                self.axs[1, 1].set_title("M22")
                self.axs[1, 0].set_xlabel("q1")
                self.axs[1, 0].set_ylabel("q2")

            if self.input_training_data[:, 0].shape[0] == self.output_training_data["N1"].shape[0]:
                self.axs[0, 2].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], self.output_training_data["N1"], '.k', markersize=1)
                self.axs[0, 2].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], output_pred_N1, '.m', markersize=1)
                self.axs[0, 2].plot3D(self.input_training_data[:, 2], self.input_training_data[:, 3], self.output_training_data["N1"], '.k', markersize=1)
                self.axs[0, 2].plot3D(self.input_training_data[:, 2], self.input_training_data[:, 3], output_pred_N1, '.r', markersize=1)
                self.axs[0, 2].set_title("N1")
                self.axs[0, 2].set_xlabel("q1 - qdot1")
                self.axs[0, 2].set_ylabel("q2 - qdot2")

            if self.input_training_data[:, 0].shape[0] == self.output_training_data["N2"].shape[0]:
                self.axs[1, 2].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], self.output_training_data["N2"], '.k', markersize=1)
                self.axs[1, 2].plot3D(self.input_training_data[:, 0], self.input_training_data[:, 1], output_pred_N2, '.m', markersize=1)
                self.axs[1, 2].plot3D(self.input_training_data[:, 2], self.input_training_data[:, 3], self.output_training_data["N2"], '.k', markersize=1)
                self.axs[1, 2].plot3D(self.input_training_data[:, 2], self.input_training_data[:, 3], output_pred_N2, '.r', markersize=1)
                self.axs[1, 2].set_title("N2")
                self.axs[1, 2].set_xlabel("q1 - qdot1")
                self.axs[1, 2].set_ylabel("q2 - qdot2")

        # Error
        if len(self.xdot_errors) > 0:
            self.axs[0, 1].plot(self.xdot_errors, '-m', label='xdot error')

            # Plot the mean error
            if len(self.xdot_errors) > 10:
                mean_errors = np.zeros((len(self.xdot_errors) - 10, ))
                for i in range(len(self.xdot_errors) - 10):
                    mean_errors[i] = np.mean(self.xdot_errors[i:i+10])
                self.axs[0, 1].plot(range(10, len(self.xdot_errors)), mean_errors, '-r', linewidth=2, label='Mean xdot error')

            if len(self.reintegration_errors) > 0:
                self.axs[0, 1].plot(self.reintegration_errors, '-c', label='Reintegration error')

                # Plot the mean error
                if len(self.reintegration_errors) > 10:
                    mean_errors = np.zeros((len(self.reintegration_errors) - 10,))
                    for i in range(len(self.reintegration_errors) - 10):
                        mean_errors[i] = np.mean(self.reintegration_errors[i:i + 10])
                    self.axs[0, 1].plot(range(10, len(self.reintegration_errors)), mean_errors, '-b', linewidth=2, label='Mean reintegration error')

            self.axs[0, 1].set_xlabel('Episode')
            self.axs[0, 1].set_title("Trajectory Error")
            self.axs[0, 1].grid(True, alpha=0.3)
            self.axs[0, 1].legend()
            self.axs[0, 1].set_yscale('log')

    def save_figure(self):
        """Save the current figure"""
        if self.fig:
            current_path = Path(__file__).parent
            spline_fig_path = f"{current_path}/../../../figures/LearningInternalDynamics/spline_parameters_learning_curve.png"
            self.fig.savefig(spline_fig_path)


class SplineParametersDynamicsLearner:
    """
    Spline approximation of the dynamics.
    """

    def __init__(self, nb_q, smoothness=0.1, kernel='thin_plate_spline', enable_plotting=True):
        self.nb_q = nb_q
        self.smoothness = smoothness
        self.kernel = kernel

        # Storage for training data
        self.input_training_data = None
        self.output_training_data = {"M11": None, "M12": None, "M22": None, "N1": None, "N2": None}

        # Spline estimates
        self.spline_model = {"M11": None, "M12": None, "M22": None, "N1": None, "N2": None}
        self.xdot_errors = []
        self.reintegration_errors = []

        # Live plots
        self.enable_plotting = enable_plotting
        self.plotter = None
        if self.enable_plotting:
            self.plotter = LivePlotter()
            self.plotter.start()

    def update(self, x_samples, M_real, N_real):
        """
        Update the spline models with new observations.

        Parameters
        ----------
        x_samples: array of shape (n_samples, nb_q * 2) - states
        M_real: array of shape (n_samples, 2, 2) - true mass matrix
        N_real: array of shape (n_samples, 2) - true non-linear effects vector
        """
        # Concatenate state and control as features
        input_new = x_samples

        # Add to training data
        if self.input_training_data is None:
            self.input_training_data = input_new
        else:
            self.input_training_data = np.vstack((self.input_training_data, input_new))

        if self.output_training_data["M11"] is None:
            self.output_training_data["M11"] = M_real[:, 0, 0]
            self.output_training_data["M12"] = M_real[:, 0, 1]
            self.output_training_data["M22"] = M_real[:, 1, 1]
            self.output_training_data["N1"] = N_real[:, 0]
            self.output_training_data["N2"] = N_real[:, 1]
        else:
            self.output_training_data["M11"] = np.hstack((self.output_training_data["M11"], M_real[:, 0, 0]))
            self.output_training_data["M12"] = np.hstack((self.output_training_data["M12"], M_real[:, 0, 1]))
            self.output_training_data["M22"] = np.hstack((self.output_training_data["M22"], M_real[:, 1, 1]))
            self.output_training_data["N1"] = np.hstack((self.output_training_data["N1"], N_real[:, 0]))
            self.output_training_data["N2"] = np.hstack((self.output_training_data["N2"], N_real[:, 1]))

        # Retrain the spline model
        for key in ["M11", "M12", "M22", "N1", "N2"]:
            self.spline_model[key] = RBFInterpolator(
                self.input_training_data,
                self.output_training_data[key],
                smoothing=self.smoothness,
                kernel=self.kernel,
            )

        # Update plotter
        if self.enable_plotting and self.plotter:
            self.plotter.update_data(
                input_data=self.input_training_data,
                output_data=self.output_training_data,
                spline_model=self.spline_model
            )

    def add_errors(self, xdot_errors, reintegration_errors):
        """Add an error measurement"""
        self.xdot_errors.append(xdot_errors)
        self.reintegration_errors.append(reintegration_errors)
        if self.enable_plotting and self.plotter:
            self.plotter.update_data(xdot_errors=xdot_errors, reintegration_errors=reintegration_errors)

    def predict(self, x, u):
        """
        Predict state derivative xdot = f(x, u) using the learned GP models.

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

        M11 = self.spline_model["M11"](input_test)[0]
        M12 = self.spline_model["M12"](input_test)[0]
        M22 = self.spline_model["M22"](input_test)[0]
        inv_mass_matrix = np.array([[M11, M12],
                                [M12, M22]])

        N1 = self.spline_model["N1"](input_test)[0]
        N2 = self.spline_model["N2"](input_test)[0]
        nonlinear_effects = np.array([N1, N2])

        dq = x[self.nb_q:].reshape(2, 1)
        dqdot = inv_mass_matrix @ np.reshape(u - nonlinear_effects, (2, 1))
        xdot_estimate = np.vstack((dq, dqdot))
        return xdot_estimate.reshape(4, )

    def forward_dyn(self, x, u, motor_noise):
        return self.predict(x, u)

    def save_model(self):
        """
        Save the learned model to a file.
        """
        current_path = Path(__file__).parent
        spline_model_path = f"{current_path}/../../../results/LearningInternalDynamics/spline_dynamics_model.pkl"

        # Stop plotter before saving
        if self.enable_plotting and self.plotter:
            self.plotter.save_figure()

        with open(spline_model_path, 'wb') as f:
            pickle.dump(self, f)

    def __del__(self):
        """Cleanup plotter on deletion"""
        if self.enable_plotting and self.plotter:
            self.plotter.stop()


def train_spline_dynamics_parameters_learner():
    np.random.seed(0)

    # Constants
    final_time = 0.8
    dt = 0.05
    nb_q = 2
    n_shooting = int(final_time / dt)

    # Get the real dynamics
    real_forward_dyn, inv_mass_matrix_func, nl_effect_vector_func = get_the_real_dynamics()

    # Initialize the Bayesian learner
    learner = SplineParametersDynamicsLearner(nb_q, enable_plotting=True)

    # Learn ten episodes
    for i_learn in range(10):

        # Generate random data to initially train on
        x0_this_time, u_this_time = generate_random_data(nb_q, n_shooting)

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

        # Update the Bayesian model with new observations
        learner.update(
            x_samples=x_integrated_real[:, :-1].T,
            M_real=M_real,
            N_real=N_real,
        )

    # Track learning progress
    for i_episode in range(1000):

        # Generate random data to compare against
        x0_this_time, u_this_time = generate_random_data(nb_q, n_shooting)

        # Evaluate the error made by the approximate dynamics
        x_integrated_approx, x_integrated_real, xdot_approx, xdot_real, M_real, N_real = integrate_the_dynamics(
            x0_this_time,
            u_this_time,
            dt,
            current_forward_dyn=learner.forward_dyn,
            real_forward_dyn=real_forward_dyn,
            inv_mass_matrix_func=inv_mass_matrix_func,
            nl_effect_vector_func=nl_effect_vector_func,
        )

        # Compute the error
        xdot_error_norm = np.linalg.norm(xdot_approx - xdot_real, axis=0) * 180 / np.pi
        xdot_errors_this_time = np.mean(xdot_error_norm)
        reintegration_error_norm = np.linalg.norm(x_integrated_approx - x_integrated_real, axis=0) * 180 / np.pi
        reintegration_errors_this_time = np.mean(reintegration_error_norm)
        learner.add_errors(xdot_errors_this_time, reintegration_errors_this_time)
        print(f"{i_episode} --- xdot error: {xdot_errors_this_time:.6f} [{np.min(xdot_error_norm)}, {np.max(xdot_error_norm)}]")

        # Update the Bayesian model with new observations
        learner.update(
            x_samples=x_integrated_real[:, :-1].T,
            M_real=M_real,
            N_real=N_real,
        )

    print("-----------------------------------------------------")
    print("Learning complete!")
    learner.save_model()

    # Keep plot open
    input("Press Enter to close...")
    learner.plotter.stop()