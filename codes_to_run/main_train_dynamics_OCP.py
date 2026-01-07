from ReachingLearning import train_spline_dynamics_parameters_ocp, evaluate_spline_dynamics_parameters_ocp

# smoothness = 0.01  # Not smooth enough
smoothness = 0.5  # Seems OK :D
# smoothness = 0.99  # Too smooth

# # nb_grid_points = 10  # Too small
# nb_grid_points = 15
# # nb_grid_points = 20  # Computer wants to explode

nb_grid_points_q = 10
nb_grid_points_qdot = 20

train_spline_dynamics_parameters_ocp(smoothness, nb_grid_points_q, nb_grid_points_qdot)

evaluate_spline_dynamics_parameters_ocp(smoothness, nb_grid_points_q, nb_grid_points_qdot)