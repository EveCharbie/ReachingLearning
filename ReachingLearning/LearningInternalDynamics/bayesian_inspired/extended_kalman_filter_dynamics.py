"""
This file was retired before even being completed.
TODO: It should be removed !
"""

import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
import biorbd_casadi as biorbd
from filterpy.kalman import ExtendedKalmanFilter
from pathlib import Path

from ...StochasticOptimalControl.utils import RK4, get_dm_value


# Optimized in Tom's version
shoulder_pos_initial = 0.349065850398866
elbow_pos_initial = 2.245867726451909
shoulder_pos_final = 0.959931088596881
elbow_pos_final = 1.159394851847144


class ExtendedKalmanFilterLearner:
    def __init__(self, real_model: biorbd.Model, n_shooting: int):

        self.n_shooting = n_shooting
        self.real_model = real_model
        self.nb_q = real_model.nbQ()

        # Initialize parameters
        q = np.zeros((self.nb_q, self.n_shooting + 1))
        q[0, :] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting + 1)  # Shoulder
        q[1, :] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting + 1)  # Elbow
        qdot = np.zeros((self.nb_q, self.n_shooting + 1))

        mass_matrix_real = np.zeros((self.nb_q, self.nb_q, self.n_shooting + 1))
        nl_effects_real = np.zeros((self.nb_q, self.n_shooting + 1))
        for i_node in range(self.n_shooting + 1):
            mass_matrix_real[:, :, i_node] = get_dm_value(real_model.massMatrix, [q[:, i_node]])
            nl_effects_real[:, i_node] = get_dm_value(real_model.NlEffects, [q[:, i_node], qdot[:, i_node]])

        self.parameters_real = np.vstack((mass_matrix_real, nl_effects_real))
        self.parameters_estimates = np.vstack((
            mass_matrix_real * np.random.rand(self.nb_q, self.nb_q, self.n_shooting + 1),
            nl_effects_real * np.random.rand(self.nb_q, self.n_shooting + 1),
        ))

    def measure_states(self, x):
        """Measurement function z = h(x)"""
        # Assume we can measure q and qdot directly and perfectly for now
        return x

    def dynamics_jacobian(self, x):
        """
        ŷ(p, t) - ŷ(^p, t) ≈ ∂ŷ/∂p(p - ^p)
        Eq 1 in Berniker & Kording 2008
        Taylor approximation to the first order
        """
        prediction_error = self.forward_dyn(self.parameters_real) - self.forward_dyn(self.parameters_estimates)
        return prediction_error

    def measurement_jacobian(self, x):
        """Jacobian of h w.r.t x"""
        return np.eye(4)

    # ---- Initialize EKF ----
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=4, dim_u=2)
    ekf.x = np.zeros(4)       # initial state [q1, q2, dq1, dq2]
    ekf.P = np.eye(4) * 0.1   # covariance
    ekf.R = np.eye(4) * 1e-3  # measurement noise
    ekf.Q = np.eye(4) * 1e-4  # process noise

    # ---- Simulation ----
    n_steps = 500
    true_state = np.zeros((n_steps, 4))
    meas = np.zeros((n_steps, 4))
    est = np.zeros((n_steps, 4))

    for t in range(1, n_steps):
        u = np.array([0.1*np.sin(0.05*t), 0.2*np.cos(0.05*t)])

        # simulate true dynamics
        true_state[t] = f(true_state[t-1], u)
        meas[t] = true_state[t] + np.random.multivariate_normal(np.zeros(4), ekf.R)

        # --- EKF predict/update ---
        F = F_jacobian(ekf.x, u)
        ekf.F = F
        ekf.predict_update(meas[t], f, h, args=(u,), hx_args=(), Fx=F_jacobian, Hx=H_jacobian)

        est[t] = ekf.x

    def forward_dyn(self, parameters: np.ndarray) -> Callable:
        # Update the dynamics function based on current parameters
        # For simplicity, assume parameters directly modify some coefficients in the dynamics
        # This is a placeholder; actual implementation depends on how parameters affect dynamics
        modified_dynamics = ...  # Modify the dynamics using parameters
        return modified_dynamics


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


def integrate_the_dynamics(x0: np.ndarray, u: np.ndarray, dt: float, current_forward_dyn, real_forward_dyn: cas.Function):

    nb_q = 2
    n_shooting = u.shape[1]
    x_integrated_approx = np.zeros((2*nb_q, n_shooting + 1))
    x_integrated_approx[:, 0] = x0
    x_integrated_real = np.zeros((2*nb_q, n_shooting + 1))
    x_integrated_real[:, 0] = x0
    xdot_real = np.zeros((2*nb_q, n_shooting))
    for i_node in range(n_shooting):

        # EKF predict/update
        F = F_jacobian(ekf.x, u)
        ekf.F = F
        ekf.predict_update(x_integrated_real[:, i_node], f, h, args=(u,), hx_args=(), Fx=F_jacobian, Hx=H_jacobian)
        current_forward_dyn = get_current_forward_dyn(parameters=ekf.x)

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

def plot_learning(errors: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Trajectory Error', fontsize=12)
    plt.title('Bayesian Dynamics Learning Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('learning_curve.png')


def train_ekf_dynamics_learner():

    np.random.seed(0)

    # Constants
    final_time = 0.8
    dt = 0.05
    nb_q = 2
    n_shooting = int(final_time / dt)

    # Get the real dynamics
    real_forward_dyn = get_the_real_dynamics()

    # Initialize the EKF learner
    learner = ExtendedKalmanFilterLearner(nb_q)

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
        x_integrated_approx, x_integrated_real, xdot_real = integrate_the_dynamics(
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


