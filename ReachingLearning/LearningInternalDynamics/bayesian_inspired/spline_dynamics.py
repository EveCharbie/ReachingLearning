import pickle
from pathlib import Path
import multiprocessing as mp
import threading

import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import casadi as cas
import biorbd_casadi as biorbd
# from scipy.interpolate import RBFInterpolator

from ...StochasticOptimalControl.utils import RK4


class LivePlotter:
    """Separate class to handle live plotting in a thread"""

    def __init__(self):
        self.fig = None
        self.axs = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

        # Data to plot
        self.input_training_data = None
        self.output_training_data = None
        self.spline_model = None
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
        self.fig, self.axs = plt.subplots(2, 3, figsize=(12, 8))
        plt.ion()

        # Samples
        self.axs[0, 0].set_title("Q samples")
        self.axs[0, 1].set_title("Qdot samples")
        self.axs[0, 2].set_title("Tau samples")

        # Predictions
        self.axs[1, 0].set_title("dQ predictions")
        self.axs[1, 1].set_title("dQdot predictions")

        # Error
        self.axs[1, 2].set_xlabel('Episode')
        self.axs[1, 2].set_title("Trajectory Error")
        self.axs[1, 2].grid(True, alpha=0.3)

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

        # Samples
        if self.input_training_data is not None:
            self.axs[0, 0].plot(self.input_training_data[:, 0], self.input_training_data[:, 1], '.k', markersize=1)
            self.axs[0, 0].set_title("Q samples")

            self.axs[0, 1].plot(self.input_training_data[:, 2], self.input_training_data[:, 3], '.k', markersize=1)
            self.axs[0, 1].set_title("Qdot samples")

            self.axs[0, 2].plot(self.input_training_data[:, 4], self.input_training_data[:, 5], '.k', markersize=1)
            self.axs[0, 2].set_title("Tau samples")

        # Predictions
        if self.output_training_data is not None and self.spline_model is not None:
            output_pred = self.spline_model(self.input_training_data)

            self.axs[1, 0].plot(self.output_training_data[:, 0], self.output_training_data[:, 1], '.k', markersize=1, label='True')
            self.axs[1, 0].plot(output_pred[:, 0], output_pred[:, 1], '.b', markersize=1, label='Predicted')
            self.axs[1, 1].plot(self.output_training_data[:, 2], self.output_training_data[:, 3], '.k', markersize=1, label='True')
            self.axs[1, 1].plot(output_pred[:, 2], output_pred[:, 3], '.b', markersize=1, label='Predicted')
            for i_episode in range(output_pred.shape[0]):
                self.axs[1, 0].plot(
                    np.array([
                    self.output_training_data[i_episode, 0],
                        output_pred[i_episode, 0]
                    ]),
                    np.array([
                        self.output_training_data[i_episode, 1],
                        output_pred[i_episode, 1]
                    ]),
                    '-r',
                    linewidth=0.5,
                )
                self.axs[1, 1].plot(
                    np.array([
                    self.output_training_data[i_episode, 2],
                        output_pred[i_episode, 2]
                    ]),
                    np.array([
                        self.output_training_data[i_episode, 3],
                        output_pred[i_episode, 3]
                    ]),
                    '-r',
                    linewidth=0.5,
                )
            self.axs[1, 0].set_title("dQ predictions")
            self.axs[1, 0].legend()
            self.axs[1, 1].set_title("dQdot predictions")
            self.axs[1, 1].legend()

        # Error
        if len(self.xdot_errors) > 0:
            self.axs[1, 2].plot(self.xdot_errors, '-m', label='xdot error')

            # Plot the mean error
            if len(self.xdot_errors) > 10:
                mean_errors = np.zeros((len(self.xdot_errors) - 10, ))
                for i in range(len(self.xdot_errors) - 10):
                    mean_errors[i] = np.mean(self.xdot_errors[i:i+10])
                self.axs[1, 2].plot(range(10, len(self.xdot_errors)), mean_errors, '-r', linewidth=2, label='Mean xdot error')

            if len(self.reintegration_errors) > 0:
                self.axs[1, 2].plot(self.reintegration_errors, '-c', label='Reintegration error')

                # Plot the mean error
                if len(self.reintegration_errors) > 10:
                    mean_errors = np.zeros((len(self.reintegration_errors) - 10,))
                    for i in range(len(self.reintegration_errors) - 10):
                        mean_errors[i] = np.mean(self.reintegration_errors[i:i + 10])
                    self.axs[1, 2].plot(range(10, len(self.reintegration_errors)), mean_errors, '-b', linewidth=2, label='Mean reintegration error')

            self.axs[1, 2].set_xlabel('Episode')
            self.axs[1, 2].set_title("Trajectory Error")
            self.axs[1, 2].grid(True, alpha=0.3)
            self.axs[1, 2].legend()
            self.axs[1, 2].set_yscale('log')

    def save_figure(self):
        """Save the current figure"""
        if self.fig:
            current_path = Path(__file__).parent
            spline_fig_path = f"{current_path}/../../../figures/LearningInternalDynamics/spline_learning_curve.pkl"
            self.fig.savefig(spline_fig_path)


class SplineDynamicsLearner:
    """
    Spline approximation of the dynamics.
    """

    def __init__(self, nb_q, smoothness=0.1, kernel='thin_plate_spline', enable_plotting=True):
        self.nb_q = nb_q
        self.smoothness = smoothness
        self.kernel = kernel

        # Storage for training data
        self.input_training_data = None
        self.output_training_data = None

        # Spline estimates
        self.spline_model = None
        self.xdot_errors = []
        self.reintegration_errors = []

        # Live plots
        self.enable_plotting = enable_plotting
        self.plotter = None
        if self.enable_plotting:
            self.plotter = LivePlotter()
            self.plotter.start()

    def update(self, x_samples, u_samples, xdot_real):
        """
        Update the spline models with new observations.

        Parameters
        ----------
        x_samples: array of shape (n_samples, nb_q * 2) - states
        u_samples: array of shape (n_samples, nb_q) - controls
        xdot_real: array of shape (n_samples, nb_q * 2) - true state derivatives
        """
        # Concatenate state and control as features
        input_new = np.hstack((x_samples, u_samples))

        # Add to training data
        if self.input_training_data is None:
            self.input_training_data = input_new
        else:
            self.input_training_data = np.vstack((self.input_training_data, input_new))

        if self.output_training_data is None:
            self.output_training_data = xdot_real
        else:
            self.output_training_data = np.vstack((self.output_training_data, xdot_real))

        # Retrain the spline model
        # self.spline_model = RBFInterpolator(
        #     self.input_training_data,
        #     self.output_training_data,
        #     smoothing=self.smoothness,
        #     kernel=self.kernel,
        # )
        self.spline_model = cas.interpolant(
            "spline_model",
            "bspline",
            self.input_training_data,
            self.output_training_data,
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
        input_test = np.hstack((x.reshape(-1, ), u.reshape(-1, ))).reshape(1, -1)
        xdot_estimate = self.spline_model(input_test)
        return xdot_estimate

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


def get_the_real_dynamics():
    current_path = Path(__file__).parent
    model_path = f"{current_path}/../../StochasticOptimalControl/models/arm_model.bioMod"
    biorbd_model = biorbd.Model(model_path)
    nb_q = biorbd_model.nbQ()
    X = cas.MX.sym("x", nb_q * 2)
    U = cas.MX.sym("u", nb_q)
    motor_noise = cas.MX.sym("motor_noise", nb_q)

    xdot = cas.vertcat(X[nb_q:], biorbd_model.ForwardDynamics(X[:nb_q], X[nb_q:], U).to_mx())
    real_dynamics = cas.Function("forward_dynamics", [X, U, motor_noise], [xdot])
    return real_dynamics


def integrate_the_dynamics(
        x0: np.ndarray,
        u: np.ndarray,
        dt: float,
        current_forward_dyn,
        real_forward_dyn: cas.Function):
    nb_q = 2
    n_shooting = u.shape[1]
    x_integrated_approx = np.zeros((2 * nb_q, n_shooting + 1))
    x_integrated_approx[:, 0] = x0
    x_integrated_real = np.zeros((2 * nb_q, n_shooting + 1))
    x_integrated_real[:, 0] = x0
    xdot_approx = np.zeros((2 * nb_q, n_shooting))
    xdot_real = np.zeros((2 * nb_q, n_shooting))
    for i_node in range(n_shooting):
        if current_forward_dyn is None:
            x_integrated_approx = None
        else:
            x_integrated_approx[:, i_node + 1] = (
                RK4(
                    x_prev=x_integrated_approx[:, i_node],
                    u=u[:, i_node],
                    dt=dt,
                    motor_noise=np.zeros((nb_q,)),
                    forward_dyn_func=current_forward_dyn,
                    n_steps=5
                )
            )[-1, :]
        x_integrated_real[:, i_node + 1] = (
            RK4(
                x_prev=x_integrated_real[:, i_node],
                u=u[:, i_node],
                dt=dt,
                motor_noise=np.zeros((nb_q,)),
                forward_dyn_func=real_forward_dyn,
                n_steps=5
            )
        )[-1, :]

        if current_forward_dyn is None:
            xdot_approx = None
        else:
            xdot_approx[:, i_node] = np.array(current_forward_dyn(
                x_integrated_real[:, i_node],
                u[:, i_node],
                np.zeros((nb_q,)),
            )).reshape(-1, )
        xdot_real[:, i_node] = np.array(real_forward_dyn(
            x_integrated_real[:, i_node],
            u[:, i_node],
            np.zeros((nb_q,)),
        )).reshape(-1, )
    return x_integrated_approx, x_integrated_real, xdot_approx, xdot_real


def generate_random_data(nb_q, n_shooting):
    # Generate random data to compare against
    x0_this_time = np.array([
        np.random.uniform(0, np.pi / 2),
        np.random.uniform(0, 7 / 8 * np.pi),
        np.random.uniform(-5, 5),
        np.random.uniform(-5, 5),
    ])
    u_this_time = np.random.uniform(-1, 1, (nb_q, n_shooting))
    return x0_this_time, u_this_time


def train_spline_dynamics_learner():
    np.random.seed(0)

    # Constants
    final_time = 0.8
    dt = 0.05
    nb_q = 2
    n_shooting = int(final_time / dt)

    # Get the real dynamics
    real_forward_dyn = get_the_real_dynamics()

    # Initialize the Bayesian learner
    learner = SplineDynamicsLearner(nb_q, enable_plotting=True)

    # Learn ten episodes
    for i_learn in range(10):

        # Generate random data to initially train on
        x0_this_time, u_this_time = generate_random_data(nb_q, n_shooting)

        # Evaluate the error made by the approximate dynamics
        _, x_integrated_real, _, xdot_real = integrate_the_dynamics(
            x0_this_time,
            u_this_time,
            dt,
            current_forward_dyn=None,
            real_forward_dyn=real_forward_dyn,
        )

        # Update the Bayesian model with new observations
        learner.update(
            x_samples=x_integrated_real[:, :-1].T,
            u_samples=u_this_time.T,
            xdot_real=xdot_real.T,
        )

    # Track learning progress
    for i_episode in range(1000):

        # Generate random data to compare against
        x0_this_time, u_this_time = generate_random_data(nb_q, n_shooting)

        # Evaluate the error made by the approximate dynamics
        x_integrated_approx, x_integrated_real, xdot_approx, xdot_real = integrate_the_dynamics(
            x0_this_time,
            u_this_time,
            dt,
            current_forward_dyn=learner.forward_dyn,
            real_forward_dyn=real_forward_dyn,
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
            u_samples=u_this_time.T,
            xdot_real=xdot_real.T,
        )

    print("-----------------------------------------------------")
    print("Learning complete!")
    learner.save_model()

    # Keep plot open
    input("Press Enter to close...")
    learner.plotter.stop()