from ReachingLearning import train_spline_dynamics_parameters_learner, evaluate_spline_dynamics_parameters

for smoothness in [0.0, 0.1, 0.5, 0.8, 0.99]:
    train_spline_dynamics_parameters_learner(smoothness)
    evaluate_spline_dynamics_parameters(smoothness)