import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
import biorbd_casadi as biorbd
import sklearn.gaussian_process as gp
from pathlib import Path

from ...StochasticOptimalControl.utils import RK4



class BayesianDynamicsLearner:
    """
    Bayesian dynamics learner using Gaussian Process regression.
    Maintains separate GP models for each state dimension.
    """

    def __init__(self, nb_q):
        self.nb_q = nb_q
        self.gp_models = []

        # Kernel
        kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.RBF(length_scale=1.0)
        self.gp_models = gp.GaussianProcessRegressor(kernel=kernel, normalize_y=True)

        # Storage for training data
        self.input_training_data = []  # q, qdot, tau
        self.output_training_data = []  # qdot, qddot

        # Prior mean function (can be zero or based on domain knowledge)
        self.use_prior = False

    def update(self, x_samples, u_samples, xdot_real):
        """
        Update the GP models with new observations.

        Parameters
        ----------
        x_samples: array of shape (n_samples, nb_q * 2) - states
        u_samples: array of shape (n_samples, nb_q) - controls
        xdot_real: array of shape (n_samples, nb_q * 2) - true state derivatives
        """
        # Concatenate state and control as features
        input_new = np.hstack((x_samples, u_samples))

        # Add to training data
        if len(self.input_training_data) == 0:
            self.input_training_data = input_new
        else:
            self.input_training_data = np.vstack((self.input_training_data, input_new))

        if len(self.output_training_data) == 0:
            self.output_training_data = xdot_real
        else:
            self.output_training_data = np.vstack((self.output_training_data, xdot_real))

        # Retrain each GP model
        self.gp_models.fit(self.input_training_data, self.output_training_data)

    def predict(self, x, u, return_std=False):
        """
        Predict state derivative xdot = f(x, u) using the learned GP models.

        Parameters
        ----------
        x: state vector of shape (2 * nb_q,)
        u: control vector of shape (nb_q,)
        return_std: if True, also return standard deviation

        Returns
        -------
        xdot_mean: predicted state derivative
        xdot_std (optional): standard deviation of prediction
        """
        # Create feature vector
        input_test = np.hstack((x.reshape(-1, ), u.reshape(-1, ))).reshape(1, -1)

        if return_std:
            mean, std = self.gp_models.predict(input_test, return_std=True)
            xdot_mean = mean[0]
            xdot_std = std[0]
            return xdot_mean, xdot_std
        else:
            xdot_mean = self.gp_models.predict(input_test)
            return xdot_mean

    def forward_dyn(self, x, u, motor_noise):
        return self.predict(x, u, return_std=False)

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


def test_the_dynamics(x0: np.ndarray, u: np.ndarray, dt: float, current_forward_dyn, real_forward_dyn: cas.Function):

    nb_q = 2
    n_shooting = u.shape[1]
    x_integrated_approx = np.zeros((2*nb_q, n_shooting + 1))
    x_integrated_approx[:, 0] = x0
    x_integrated_real = np.zeros((2*nb_q, n_shooting + 1))
    x_integrated_real[:, 0] = x0
    xdot_real = np.zeros((2*nb_q, n_shooting))
    for i_node in range(n_shooting):
        x_integrated_approx[:, i_node + 1] = (
            RK4(
                x_prev=x_integrated_approx[:, i_node],
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

def plot_learning(errors: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Trajectory Error', fontsize=12)
    plt.title('Bayesian Dynamics Learning Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('learning_curve.png')


def train_bayesian_dynamics_learner():

    np.random.seed(0)

    # Constants
    final_time = 0.8
    dt = 0.05
    nb_q = 2
    n_shooting = int(final_time / dt)

    # Get the real dynamics
    real_forward_dyn = get_the_real_dynamics()

    # Initialize the Bayesian learner
    learner = BayesianDynamicsLearner(nb_q)

    # Track learning progress
    errors = []

    for i_episode in range(100):

        # Generate random data to compare against
        x0_this_time = np.array([
            np.random.uniform(0, np.pi/2),
            np.random.uniform(0, 7/8 * np.pi),
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            ])
        u_this_time = np.random.uniform(-5, 5, (nb_q, n_shooting))

        # Evaluate the error made by the approximate dynamics
        x_integrated_approx, x_integrated_real, xdot_real = test_the_dynamics(
            x0_this_time,
            u_this_time,
            dt,
            current_forward_dyn=learner.forward_dyn,
            real_forward_dyn=real_forward_dyn,
        )

        # Compute the error
        errors_this_time = np.mean(np.linalg.norm(x_integrated_approx - x_integrated_real, axis=1))
        errors.append(errors_this_time)
        print(f"{i_episode} --- Trajectory error: {errors_this_time:.6f}")

        # Update the Bayesian model with new observations
        learner.update(
            x_samples=x_integrated_real[:, :-1].T,
            u_samples=u_this_time.T,
            xdot_real=xdot_real.T,
        )

        # Print progress every 10 episodes
        if (i_episode) % 10 == 0:
            recent_error = np.mean(errors[-10:])
            print(f"  Average error episodes : {recent_error:.6f}")

    print("-----------------------------------------------------")
    print("Learning complete!")
    plot_learning(np.array(errors))


