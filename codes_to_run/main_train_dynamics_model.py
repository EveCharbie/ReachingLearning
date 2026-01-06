import numpy as np

from ReachingLearning import train_spline_dynamics_parameters_learner, evaluate_spline_dynamics_parameters


for smoothness in [0.0]: # , 0.1, 0.5, 0.8, 0.99]:
    train_spline_dynamics_parameters_learner(smoothness, enable_plotting=True, max_tau=1, max_velocity=10*np.pi)
    evaluate_spline_dynamics_parameters(smoothness)




"""
max_tau = 10 & max_velocity = 10*np.pi -> Bad results
max_tau = 10 & max_velocity = 5*np.pi -> Not bad, but not good results
max_tau = 10 & max_velocity = 5 -> Good results

max_tau = 5 & max_velocity = 10*np.pi -> Bad results
max_tau = 1 & max_velocity = 10*np.pi -> Bad results
"""