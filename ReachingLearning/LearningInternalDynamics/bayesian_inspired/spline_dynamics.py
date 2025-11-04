import pickle
from pathlib import Path
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import casadi as cas
import biorbd_casadi as biorbd
from scipy.interpolate import RBFInterpolator

from ...StochasticOptimalControl.utils import RK4


class SplineDynamicsLearner:
    """
    Spline approximation of the dynamics.
    """

    def __init__(self, nb_q, smoothness=0.1, kernel='thin_plate_spline'):
        self.nb_q = nb_q
        self.smoothness = smoothness
        self.kernel = kernel

        # Storage for training data
        self.input_training_data = None
        self.output_training_data = None

        # Spline estimates
        self.spline_model = None
        self.errors = []

        # Live plots
        self.axs = None
        self.fig = None
        self.queue = mp.Queue()
        self.plotter = LivePlotter()
        self.plot_process = mp.Process(
            target=self.plotter, args=(), daemon=True
        )
        self.plot_process.start()

    def initialize_plots(self):
        fig, axs = plt.subplots(2, 3, figsize=(10, 10))

        # Samples
        axs[0, 0].plot(0, 0, '.k')
        axs[0, 0].set_title("Q samples")
        axs[0, 1].plot(0, 0, '.k')
        axs[0, 1].set_title("Qdot samples")
        axs[0, 2].plot(0, 0, '.k')
        axs[0, 2].set_title("Tau samples")

        # Predictions
        axs[1, 0].plot(0, 0, '.k')
        axs[1, 1].plot(0, 0, '.k')
        axs[1, 0].plot(0, 0, '.b')
        axs[1, 1].plot(0, 0, '.b')
        axs[1, 0].plot(0, 0, '-r')
        axs[1, 1].plot(0, 0, '-r')
        axs[1, 0].set_title("dQ predictions")
        axs[1, 1].set_title("dQdot predictions")

        # Error
        axs[1, 2].plot(0, 0, '-r')
        axs[1, 2].set_xlabel('Episode')
        axs[1, 2].set_title("Trajectory Error")
        axs[1, 2].grid(True, alpha=0.3)

        self.axs = axs
        self.fig = fig

    def update_plots(self):

        # Samples
        if self.input_training_data is not None and self.axs is not None:
            self.axs[0, 0].cla()
            self.axs[0, 0].plot(self.input_training_data[:, 0], self.input_training_data[:, 1], '.k')
            self.axs[0, 0].set_title("Q samples")
            self.axs[0, 1].cla()
            self.axs[0, 1].plot(self.input_training_data[:, 2], self.input_training_data[:, 3], '.k')
            self.axs[0, 1].set_title("Qdot samples")
            self.axs[0, 2].cla()
            self.axs[0, 2].plot(self.input_training_data[:, 4], self.input_training_data[:, 5], '.k')
            self.axs[0, 2].set_title("Tau samples")

        # Predictions
        if self.output_training_data is not None and self.spline_model is not None and self.axs is not None:
            self.axs[1, 0].cla()
            self.axs[1, 1].cla()
            self.axs[1, 0].plot(self.output_training_data[:, 0], self.output_training_data[:, 1], '.k', label='True')
            self.axs[1, 1].plot(self.output_training_data[:, 2], self.output_training_data[:, 3], '.k', label='True')
            output_pred = self.spline_model(self.input_training_data)
            self.axs[1, 0].plot(output_pred[:, 0], output_pred[:, 1], '.b', label='Predicted')
            self.axs[1, 1].plot(output_pred[:, 2], output_pred[:, 3], '.b', label='Predicted')
            for i_episoede in range(self.input_training_data.shape[0]):
                self.axs[1, 0].plot(
                    [self.output_training_data[:, 0], output_pred[:, 0]],
                    [self.output_training_data[:, 1], output_pred[:, 1]],
                    '-r',
                    label='Error',
                )
                self.axs[1, 1].plot(
                    [self.output_training_data[:, 2], output_pred[:, 2]],
                    [self.output_training_data[:, 3], output_pred[:, 3]],
                    '-r',
                    label='Error',
                )
            self.axs[1, 0].set_title("dQ predictions")
            self.axs[1, 1].set_title("dQdot predictions")
            # self.axs[0, 1].legend()
            # self.axs[1, 1].legend()

        # Error
        if self.axs is not None:
            self.axs[1, 2].plot(self.errors, '-r')

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

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
        self.spline_model = RBFInterpolator(
            self.input_training_data,
            self.output_training_data,
            smoothing=self.smoothness,
            kernel=self.kernel,
        )

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
        with open(spline_model_path, 'wb') as f:
            pickle.dump(self, f)

        plt.savefig('learning_curve.png')


class LivePlotter:

    def __init__(self, learner: SplineDynamicsLearner = None):
        self.learner = learner

    def __call__(self, pipe: mp.Queue, options: dict):
        """
        Parameters
        ----------
        pipe: mp.Queue
            The multiprocessing queue to evaluate
        options: dict
            The option to pass
        """
        self.pipe = pipe
        self.learner.initialize_plots()
        timer = self.learner.fig.canvas.new_timer(interval=100)
        timer.add_callback(self.callback)
        timer.start()
        plt.show()

    def callback(self) -> bool:
        """
        The callback to update the graphs

        Returns
        -------
        True if everything went well
        """

        while not self.pipe.empty():
            args = self.pipe.get()
            self.learner.update_plots()

        self.learner.fig.canvas.draw()
        self.learner.fig.canvas.flush_events()
        return True



def get_the_real_dynamics():

    current_path = Path(__file__).parent
    model_path = f"{current_path}/../../StochasticOptimalControl/models/arm_model.bioMod"
    biorbd_model = biorbd.Model(model_path)
    nb_q = biorbd_model.nbQ()
    X = cas.MX.sym("x", nb_q * 2)
    U = cas.MX.sym("u", nb_q)
    motor_noise = cas.MX.sym("motor_noise", nb_q)

    # TODO: get to the muscle dynamics
    # def get_muscle_torque(self, q: cas.MX, qdot: cas.MX, mus_activations: cas.MX) -> cas.MX:
    #     muscles_states = self.biorbd_model.stateSet()
    #     for k in range(self.biorbd_model.nbMuscles()):
    #         muscles_states[k].setActivation(mus_activations[k])
    #     q_biorbd = biorbd.GeneralizedCoordinates(q)
    #     qdot_biorbd = biorbd.GeneralizedVelocity(qdot)
    #     return self.biorbd_model.muscularJointTorque(muscles_states, q_biorbd, qdot_biorbd).to_mx()

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
    x_integrated_approx = np.zeros((2*nb_q, n_shooting + 1))
    x_integrated_approx[:, 0] = x0
    x_integrated_real = np.zeros((2*nb_q, n_shooting + 1))
    x_integrated_real[:, 0] = x0
    xdot_real = np.zeros((2*nb_q, n_shooting))
    for i_node in range(n_shooting):
        if current_forward_dyn is None:
            x_integrated_approx = None
        else:
            x_integrated_approx[:, i_node + 1] = (
                RK4(
                    x_prev=x_integrated_real[:, i_node],
                    u=u[:, i_node],
                    dt=dt,
                    motor_noise=np.zeros((nb_q, )),
                    forward_dyn_func=current_forward_dyn,
                    n_steps=5
                )
            )[-1, :]
        x_integrated_real[:, i_node + 1] = (
            RK4(
                x_prev=x_integrated_real[:, i_node],
                u=u[:, i_node],
                dt=dt,
                motor_noise=np.zeros((nb_q, )),
                forward_dyn_func=real_forward_dyn,
                n_steps=5
            )
        )[-1, :]
        xdot_real[:, i_node] = np.array(real_forward_dyn(
            x_integrated_real[:, i_node],
            u[:, i_node],
            np.zeros((nb_q,)),
        )).reshape(-1, )
    return x_integrated_approx, x_integrated_real, xdot_real

def generate_random_data(nb_q, n_shooting):
    # Generate random data to compare against
    x0_this_time = np.array([
        np.random.uniform(0, np.pi / 2),
        np.random.uniform(0, 7 / 8 * np.pi),
        np.random.uniform(-5, 5),
        np.random.uniform(-5, 5),
    ])
    u_this_time = np.random.uniform(-5, 5, (nb_q, n_shooting))
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
    learner = SplineDynamicsLearner(nb_q)

    # Learn ten episodes
    for i_learn in range(10):

        x0_this_time, u_this_time = generate_random_data(nb_q, n_shooting)

        # Evaluate the error made by the approximate dynamics
        _, x_integrated_real, xdot_real = integrate_the_dynamics(
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
        x0_this_time = np.array([
            np.random.uniform(0, np.pi/2),
            np.random.uniform(0, 7/8 * np.pi),
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            ])
        u_this_time = np.random.uniform(-5, 5, (nb_q, n_shooting))

        # Evaluate the error made by the approximate dynamics
        x_integrated_approx, x_integrated_real, xdot_real = integrate_the_dynamics(
            x0_this_time,
            u_this_time,
            dt,
            current_forward_dyn=learner.forward_dyn,
            real_forward_dyn=real_forward_dyn,
        )

        # Compute the error
        errors_this_time = np.mean(np.linalg.norm(x_integrated_approx - x_integrated_real, axis=1))
        learner.errors.append(errors_this_time)
        print(f"{i_episode} --- Trajectory error: {errors_this_time:.6f}")

        if i_episode % 10 == 0:
            learner.update_plots()

        # Update the Bayesian model with new observations
        learner.update(
            x_samples=x_integrated_real[:, :-1].T,
            u_samples=u_this_time.T,
            xdot_real=xdot_real.T,
        )

    print("-----------------------------------------------------")
    print("Learning complete!")
    learner.save_model()


