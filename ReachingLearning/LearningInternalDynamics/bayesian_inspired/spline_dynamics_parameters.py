import pickle
from pathlib import Path
import threading
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator


from .utils import (
    get_the_real_dynamics,
    integrate_the_dynamics,
    generate_random_data,
)


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
            time.sleep(0.1)

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

    def save_figure(self, sup_str: str = ""):
        """Save the current figure"""
        if self.fig:
            self.running = False
            if self.thread:
                self.thread.join()

            # Actually save the figure
            current_path = Path(__file__).parent
            spline_fig_path = f"{current_path}/../../../figures/LearningInternalDynamics/spline_parameters_learning_curve_{sup_str}.png"
            self.fig.savefig(spline_fig_path)


class SplineParametersDynamicsLearner:
    """
    Spline approximation of the dynamics.
    """

    def __init__(
            self,
            nb_q: int,
            smoothness: float = 0.1,
            kernel: str = 'thin_plate_spline',
            enable_plotting: bool = True,
    ):
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

    def save_model(self, str_sup: str = ""):
        """
        Save the learned model to a file.
        """
        current_path = Path(__file__).parent
        spline_model_path = f"{current_path}/../../../results/LearningInternalDynamics/spline_dynamics_model_{str_sup}.pkl"

        # Stop plotter before saving
        if self.enable_plotting and self.plotter:
            self.plotter.save_figure(sup_str=str_sup)

        with open(spline_model_path, 'wb') as f:
            output = {
                "input_training_data": self.input_training_data,
                "output_training_data": self.output_training_data,
                "spline_model": self.spline_model,
                "xdot_errors": self.xdot_errors,
                "reintegration_errors": self.reintegration_errors,
            }
            pickle.dump(output, f)

    def __del__(self):
        """Cleanup plotter on deletion"""
        if self.enable_plotting and self.plotter:
            self.plotter.stop()

def get_tau_opt(q: np.ndarray, qdot: np.ndarray, muscles: np.ndarray) -> np.ndarray:
    import biorbd
    current_path = Path(__file__).parent
    model_path = f"{current_path}/../../StochasticOptimalControl/models/arm_model.bioMod"
    biorbd_model = biorbd.Biorbd(model_path)

    tau_opt = np.zeros((q.shape[0], muscles.shape[1]))
    for i_node in range(muscles.shape[1]):
        tau_opt[:, i_node] = biorbd_model.muscles.joint_torque(activations=muscles[:, i_node], q=q[:, i_node], qdot=qdot[:, i_node])
    return tau_opt

def train_spline_dynamics_parameters_learner(smoothness: float, enable_plotting: bool, max_tau: float = 10.0, max_velocity: float = 10 * np.pi):
    np.random.seed(0)

    # Constants
    final_time = 0.8
    dt = 0.05
    nb_q = 2
    n_shooting = int(final_time / dt)

    # Get the real dynamics
    real_forward_dyn, inv_mass_matrix_func, nl_effect_vector_func = get_the_real_dynamics()

    # Initialize the Spline parameter learner
    learner = SplineParametersDynamicsLearner(nb_q, smoothness, enable_plotting=enable_plotting)

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
    output_file_path = f"{current_path}/../../../results/LearningInternalDynamics/spline_dynamics_parameters_{str(smoothness).replace('.', 'p')}.txt"
    output_file = open(output_file_path, 'w')
    sys.stdout = output_file

    # Track learning progress
    for i_episode in range(100):

        # Generate random data to compare against
        x0_this_time, u_this_time = generate_random_data(nb_q, n_shooting, max_tau=max_tau, max_velocity=max_velocity)

        # Evaluate the error made by the approximate dynamics
        current_forward_dyn = learner.forward_dyn
        x_integrated_approx, x_integrated_real, xdot_approx, xdot_real, M_real, N_real = integrate_the_dynamics(
            x0_this_time,
            u_this_time,
            dt,
            current_forward_dyn=current_forward_dyn,
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
        print(f"{i_episode} --- reintegration error: {reintegration_errors_this_time:.6f} [{np.min(reintegration_error_norm)}, {np.max(reintegration_error_norm)}] deg")

        # Update the Spline parameter model with new observations
        learner.update(
            x_samples=x_integrated_real[:, :-1].T,
            M_real=M_real,
            N_real=N_real,
        )

    print("-----------------------------------------------------")
    print("Learning complete!")
    learner.save_model(str_sup=f"{str(smoothness).replace('.', 'p')}")
    if enable_plotting:
        learner.plotter.stop()

    # Close the file and restore printing to the console
    sys.stdout = sys.__stdout__
    output_file.close()


def evaluate_spline_dynamics_parameters(smoothness: float = 0.1):
    """Test loading the saved model"""

    # Load the model
    current_path = Path(__file__).parent
    spline_model_path = f"{current_path}/../../../results/LearningInternalDynamics/spline_dynamics_model_{str(smoothness).replace('.', 'p')}.pkl"
    with open(spline_model_path, 'rb') as f:
        learned_parameters = pickle.load(f)
    learner = SplineParametersDynamicsLearner(nb_q=2, smoothness=smoothness, enable_plotting=False).from_learned_parameters(learned_parameters)

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
    fig_path = f"{current_path}/../../../figures/LearningInternalDynamics/spline_dynamics_model_evaluation_{str(smoothness).replace('.', 'p')}.png"
    plt.savefig(fig_path)

