from ReachingLearning import train_spline_dynamics_parameters_ocp

# smoothness = 0.01  # Not smooth enough
smoothness = 0.5  # Seems OK :D
# smoothness = 0.99  # Too smooth

smoothness = 0.01

train_spline_dynamics_parameters_ocp(smoothness)