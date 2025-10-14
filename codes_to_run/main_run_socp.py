from datetime import datetime

from ReachingLearning import (
    ExampleType,
    run_ocp,
    run_socp_basic,
)


RUN_OCP = True
RUN_SOCP_BASIC = True
print(RUN_OCP, RUN_SOCP_BASIC)
print(datetime.now().strftime("%d-%m %H:%M:%S"))

PLOT_FLAG = True
ANIMATE_FLAG = True
example_type = ExampleType.CIRCLE
force_field_magnitude = 0


n_random = 5
n_threads = 12
n_simulations = 30
seed = 0

n_q = 2
dt = 0.01
final_time = 0.8
n_shooting = int(final_time / dt)
tol = 1e-6

motor_noise_std = 0.01
wPq_std = 3e-4
wPqdot_std = 0.0024

# solver.set_bound_frac(1e-8)
# solver.set_bound_push(1e-8)


# # --- Run optimizations --- #
# run_ocp(
#     final_time=final_time,
#     n_shooting=n_shooting,
#     motor_noise_std=motor_noise_std,
#     force_field_magnitude=force_field_magnitude,
#     example_type=example_type,
#     n_threads=n_threads,
#     tol=tol,
#     n_simulations=n_simulations,
#     RUN_OCP=RUN_OCP,
#     PLOT_FLAG=PLOT_FLAG,
#     ANIMATE_FLAG=ANIMATE_FLAG,
# )

run_socp_basic(
    final_time=final_time,
    n_shooting=n_shooting,
    motor_noise_std=motor_noise_std,
    wPq_std=wPq_std,
    wPqdot_std=wPqdot_std,
    force_field_magnitude=force_field_magnitude,
    example_type=example_type,
    n_random=n_random,
    seed=seed,
    n_threads=n_threads,
    tol=tol,
    n_simulations=n_simulations,
    RUN_SOCP_BASIC=RUN_SOCP_BASIC,
    PLOT_FLAG=PLOT_FLAG,
    ANIMATE_FLAG=ANIMATE_FLAG,
)

# #
# # # --- Run the SOCP+ (variable noise) --- #
# # if RUN_SOCP_VARIABLE:
# #     save_path = save_path.replace(".pkl", "_VARIABLE.pkl")
# #
# #     motor_noise_magnitude = cas.DM(
# #         np.array(
# #             [
# #                 motor_noise_std**2 / dt,
# #                 motor_noise_std**2 / dt,
# #                 motor_noise_std**2 / dt,
# #                 motor_noise_std**2 / dt,
# #             ]
# #         )
# #     )  # All DoFs except root
# #     sensory_noise_magnitude = cas.DM(
# #         np.array(
# #             [
# #                 wPq_std**2 / dt,  # Proprioceptive position
# #                 wPq_std**2 / dt,
# #                 wPq_std**2 / dt,
# #                 wPq_std**2 / dt,
# #                 wPqdot_std**2 / dt,  # Proprioceptive velocity
# #                 wPqdot_std**2 / dt,
# #                 wPqdot_std**2 / dt,
# #                 wPqdot_std**2 / dt,
# #                 wPq_std**2 / dt,  # Vestibular position
# #                 wPq_std**2 / dt,  # Vestibular velocity
# #             ]
# #         )
# #     )
# #
# #     path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
# #     with open(path_to_results, "rb") as file:
# #         data = pickle.load(file)
# #         q_roots_last = data["q_roots_sol"]
# #         q_joints_last = data["q_joints_sol"]
# #         qdot_roots_last = data["qdot_roots_sol"]
# #         qdot_joints_last = data["qdot_joints_sol"]
# #         tau_joints_last = data["tau_joints_sol"]
# #         time_last = data["time_sol"]
# #         k_last = None
# #         ref_last = None
# #
# #     motor_noise_numerical, sensory_noise_numerical, socp, noised_states = prepare_socp_VARIABLE(
# #         biorbd_model_path=biorbd_model_path,
# #         time_last=time_last,
# #         n_shooting=n_shooting,
# #         motor_noise_magnitude=motor_noise_magnitude,
# #         sensory_noise_magnitude=sensory_noise_magnitude,
# #         q_roots_last=q_roots_last,
# #         q_joints_last=q_joints_last,
# #         qdot_roots_last=qdot_roots_last,
# #         qdot_joints_last=qdot_joints_last,
# #         tau_joints_last=tau_joints_last,
# #         k_last=None,
# #         ref_last=None,
# #         n_random=n_random,
# #     )
# #
# #     socp.add_plot_penalty()
# #     # socp.add_plot_check_conditioning()
# #     socp.add_plot_ipopt_outputs()
# #
# #     solver.set_tol(tol)
# #     sol_socp = socp.solve(solver)
# #
# #     states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
# #     controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
# #
# #     q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
# #         states["q_roots"],
# #         states["q_joints"],
# #         states["qdot_roots"],
# #         states["qdot_joints"],
# #     )
# #     tau_joints_sol, k_sol, ref_sol = controls["tau_joints"], controls["k"], controls["ref"]
# #     time_sol = sol_socp.decision_time()[-1]
# #
# #     data = {
# #         "q_roots_sol": q_roots_sol,
# #         "q_joints_sol": q_joints_sol,
# #         "qdot_roots_sol": qdot_roots_sol,
# #         "qdot_joints_sol": qdot_joints_sol,
# #         "tau_joints_sol": tau_joints_sol,
# #         "time_sol": time_sol,
# #         "k_sol": k_sol,
# #         "ref_sol": ref_sol,
# #         "motor_noise_numerical": motor_noise_numerical,
# #         "sensory_noise_numerical": sensory_noise_numerical,
# #     }
# #
# #     save_path = save_path.replace(".", "p")
# #     if sol_socp.status != 0:
# #         save_path = save_path.replace("ppkl", f"_DMS_{n_random}random_DVG_{print_tol}.pkl")
# #     else:
# #         save_path = save_path.replace("ppkl", f"_DMS_{n_random}random_CVG_{print_tol}.pkl")
# #
# #     # --- Save the results --- #
# #     with open(save_path, "wb") as file:
# #         pickle.dump(data, file)
# #
# #     with open(save_path.replace(".pkl", f"_sol.pkl"), "wb") as file:
# #         del sol_socp.ocp
# #         pickle.dump(sol_socp, file)
# #
# #     print(save_path)
# #     # import bioviz
# #     # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
# #     # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
# #     # b.exec()
# #
# #
# # # --- Run the SOCP+ (feedforward) --- #
# # n_q += 1
# #
# # if RUN_SOCP_FEEDFORWARD:
# #     save_path = save_path.replace(".pkl", "_FEEDFORWARD.pkl")
# #
# #     motor_noise_magnitude = cas.DM(
# #         np.array(
# #             [
# #                 motor_noise_std**2 / dt,
# #                 0.0,
# #                 motor_noise_std**2 / dt,
# #                 motor_noise_std**2 / dt,
# #                 motor_noise_std**2 / dt,
# #             ]
# #         )
# #     )  # All DoFs except root
# #     sensory_noise_magnitude = cas.DM(
# #         np.array(
# #             [
# #                 wPq_std**2 / dt,  # Proprioceptive position
# #                 wPq_std**2 / dt,
# #                 wPq_std**2 / dt,
# #                 wPq_std**2 / dt,
# #                 wPqdot_std**2 / dt,  # Proprioceptive velocity
# #                 wPqdot_std**2 / dt,
# #                 wPqdot_std**2 / dt,
# #                 wPqdot_std**2 / dt,
# #                 wPq_std**2 / dt,  # Vestibular position
# #                 wPq_std**2 / dt,  # Vestibular velocity
# #                 wPq_std**2 / dt,  # Visual
# #             ]
# #         )
# #     )
# #
# #     path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
# #     with open(path_to_results, "rb") as file:
# #         data = pickle.load(file)
# #         q_roots_last = data["q_roots_sol"]
# #         q_joints_last = data["q_joints_sol"]
# #         qdot_roots_last = data["qdot_roots_sol"]
# #         qdot_joints_last = data["qdot_joints_sol"]
# #         tau_joints_last = data["tau_joints_sol"]
# #         time_last = data["time_sol"]
# #         k_last = None
# #         ref_last = None
# #
# #     q_joints_last = np.vstack((q_joints_last[0, :], np.zeros((1, q_joints_last.shape[1])), q_joints_last[1:, :]))
# #     q_joints_last[1, :5] = -0.5
# #     q_joints_last[1, 5:-5] = np.linspace(-0.5, 0.3, n_shooting + 1 - 10)
# #     q_joints_last[1, -5:] = 0.3
# #
# #     qdot_joints_last = np.vstack(
# #         (qdot_joints_last[0, :], np.ones((1, qdot_joints_last.shape[1])) * 0.01, qdot_joints_last[1:, :])
# #     )
# #     tau_joints_last = np.vstack(
# #         (tau_joints_last[0, :], np.ones((1, tau_joints_last.shape[1])) * 0.01, tau_joints_last[1:, :])
# #     )
# #
# #     motor_noise_numerical, sensory_noise_numerical, socp, noised_states = prepare_socp_FEEDFORWARD(
# #         biorbd_model_path=biorbd_model_path_vision,
# #         time_last=time_last,
# #         n_shooting=n_shooting,
# #         motor_noise_magnitude=motor_noise_magnitude,
# #         sensory_noise_magnitude=sensory_noise_magnitude,
# #         q_roots_last=q_roots_last,
# #         q_joints_last=q_joints_last,
# #         qdot_roots_last=qdot_roots_last,
# #         qdot_joints_last=qdot_joints_last,
# #         tau_joints_last=tau_joints_last,
# #         k_last=None,
# #         ref_last=None,
# #         n_random=n_random,
# #     )
# #
# #     socp.add_plot_penalty()
# #     socp.add_plot_ipopt_outputs()
# #
# #     solver.set_tol(tol)
# #     sol_socp = socp.solve(solver)
# #
# #     states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
# #     controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
# #
# #     q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
# #         states["q_roots"],
# #         states["q_joints"],
# #         states["qdot_roots"],
# #         states["qdot_joints"],
# #     )
# #     tau_joints_sol, k_sol, ref_fb_sol = controls["tau_joints"], controls["k"], controls["ref"]
# #     time_sol = sol_socp.decision_time()[-1]
# #     ref_ff_sol = sol_socp.parameters["final_somersault"]
# #
# #     data = {
# #         "q_roots_sol": q_roots_sol,
# #         "q_joints_sol": q_joints_sol,
# #         "qdot_roots_sol": qdot_roots_sol,
# #         "qdot_joints_sol": qdot_joints_sol,
# #         "tau_joints_sol": tau_joints_sol,
# #         "time_sol": time_sol,
# #         "k_sol": k_sol,
# #         "ref_fb_sol": ref_fb_sol,
# #         "ref_ff_sol": ref_ff_sol,  # final somersault
# #         "motor_noise_numerical": motor_noise_numerical,
# #         "sensory_noise_numerical": sensory_noise_numerical,
# #     }
# #
# #     save_path = save_path.replace(".", "p")
# #     if sol_socp.status != 0:
# #         save_path = save_path.replace("ppkl", f"_DMS_{n_random}random_DVG_{print_tol}.pkl")
# #     else:
# #         save_path = save_path.replace("ppkl", f"_DMS_{n_random}random_CVG_{print_tol}.pkl")
# #
# #     # --- Save the results --- #
# #     with open(save_path, "wb") as file:
# #         pickle.dump(data, file)
# #
# #     with open(save_path.replace(".pkl", f"_sol.pkl"), "wb") as file:
# #         del sol_socp.ocp
# #         pickle.dump(sol_socp, file)
# #
# #     print(save_path)
# #     # import bioviz
# #     # b = bioviz.Viz(model_path=biorbd_model_path_vision_with_mesh)
# #     # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
# #     # b.exec()
# #
# #
# # # --- Run the SOCP+ (variable noise & feedforward) --- #
# # save_path = save_path.replace(".pkl", "_VARIABLE_FEEDFORWARD.pkl")
# # n_q += 1
# #
# # if RUN_SOCP_VARIABLE_FEEDFORWARD:
# #
# #     motor_noise_magnitude = cas.DM(
# #         np.array(
# #             [
# #                 motor_noise_std**2 / dt,
# #                 0.0,
# #                 motor_noise_std**2 / dt,
# #                 motor_noise_std**2 / dt,
# #                 motor_noise_std**2 / dt,
# #             ]
# #         )
# #     )  # All DoFs except root
# #     sensory_noise_magnitude = cas.DM(
# #         np.array(
# #             [
# #                 wPq_std**2 / dt,  # Proprioceptive position
# #                 wPq_std**2 / dt,
# #                 wPq_std**2 / dt,
# #                 wPq_std**2 / dt,
# #                 wPqdot_std**2 / dt,  # Proprioceptive velocity
# #                 wPqdot_std**2 / dt,
# #                 wPqdot_std**2 / dt,
# #                 wPqdot_std**2 / dt,
# #                 wPq_std**2 / dt,  # Vestibular position
# #                 wPq_std**2 / dt,  # Vestibular velocity
# #                 wPq_std**2 / dt,  # Visual
# #             ]
# #         )
# #     )
# #
# #     path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
# #     with open(path_to_results, "rb") as file:
# #         data = pickle.load(file)
# #         q_roots_last = data["q_roots_sol"]
# #         q_joints_last = data["q_joints_sol"]
# #         qdot_roots_last = data["qdot_roots_sol"]
# #         qdot_joints_last = data["qdot_joints_sol"]
# #         tau_joints_last = data["tau_joints_sol"]
# #         time_last = data["time_sol"]
# #         k_last = None
# #         ref_last = None
# #         ref_ff_last = None
# #     q_joints_last = np.vstack((q_joints_last[0, :], np.zeros((1, q_joints_last.shape[1])), q_joints_last[1:, :]))
# #     q_joints_last[1, :5] = -0.5
# #     q_joints_last[1, 5:-5] = np.linspace(-0.5, 0.3, n_shooting + 1 - 10)
# #     q_joints_last[1, -5:] = 0.3
# #
# #     qdot_joints_last = np.vstack(
# #         (qdot_joints_last[0, :], np.ones((1, qdot_joints_last.shape[1])) * 0.01, qdot_joints_last[1:, :])
# #     )
# #     tau_joints_last = np.vstack(
# #         (tau_joints_last[0, :], np.ones((1, tau_joints_last.shape[1])) * 0.01, tau_joints_last[1:, :])
# #     )
# #     motor_noise_numerical, sensory_noise_numerical, socp, noised_states = prepare_socp_VARIABLE_FEEDFORWARD(
# #         biorbd_model_path=biorbd_model_path_vision,
# #         time_last=time_last,
# #         n_shooting=n_shooting,
# #         motor_noise_magnitude=motor_noise_magnitude,
# #         sensory_noise_magnitude=sensory_noise_magnitude,
# #         q_roots_last=q_roots_last,
# #         q_joints_last=q_joints_last,
# #         qdot_roots_last=qdot_roots_last,
# #         qdot_joints_last=qdot_joints_last,
# #         tau_joints_last=tau_joints_last,
# #         k_last=k_last,
# #         ref_last=ref_last,
# #         ref_ff_last=ref_ff_last,
# #         n_random=n_random,
# #     )
# #     socp.add_plot_penalty()
# #     socp.add_plot_ipopt_outputs()
# #
# #     save_path = save_path.replace(".", "p")
# #
# #     date_time = datetime.now().strftime("%d-%m-%H-%M-%S")
# #     path_to_temporary_results = f"temporary_results_{date_time}"
# #     if path_to_temporary_results not in os.listdir("results/"):
# #         os.mkdir("results/" + path_to_temporary_results)
# #     nb_iter_save = 10
# #     # sol_last.ocp.save_intermediary_ipopt_iterations(
# #     #     "results/" + path_to_temporary_results, "Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_VARIABLE_FEEDFORWARD", nb_iter_save
# #     # )
# #     socp.save_intermediary_ipopt_iterations(
# #         "results/" + path_to_temporary_results,
# #         "Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_VARIABLE_FEEDFORWARD",
# #         nb_iter_save,
# #     )
# #
# #     solver.set_tol(tol)
# #     sol_socp = socp.solve(solver)
# #
# #     states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
# #     controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
# #
# #     q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
# #         states["q_roots"],
# #         states["q_joints"],
# #         states["qdot_roots"],
# #         states["qdot_joints"],
# #     )
# #     tau_joints_sol, k_sol, ref_fb_sol = controls["tau_joints"], controls["k"], controls["ref"]
# #     time_sol = sol_socp.decision_time()[-1]
# #     ref_ff_sol = sol_socp.parameters["final_somersault"]
# #
# #     data = {
# #         "q_roots_sol": q_roots_sol,
# #         "q_joints_sol": q_joints_sol,
# #         "qdot_roots_sol": qdot_roots_sol,
# #         "qdot_joints_sol": qdot_joints_sol,
# #         "tau_joints_sol": tau_joints_sol,
# #         "time_sol": time_sol,
# #         "k_sol": k_sol,
# #         "ref_fb_sol": ref_fb_sol,
# #         "ref_ff_sol": ref_ff_sol,
# #         "motor_noise_numerical": motor_noise_numerical,
# #         "sensory_noise_numerical": sensory_noise_numerical,
# #     }
# #
# #     if sol_socp.status != 0:
# #         save_path = save_path.replace("ppkl", f"_DMS_{n_random}random_DVG_{print_tol}.pkl")
# #     else:
# #         save_path = save_path.replace("ppkl", f"_DMS_{n_random}random_CVG_{print_tol}.pkl")
# #
# #     # --- Save the results --- #
# #     with open(save_path, "wb") as file:
# #         pickle.dump(data, file)
# #
# #     with open(save_path.replace(".pkl", f"_sol.pkl"), "wb") as file:
# #         del sol_socp.ocp
# #         pickle.dump(sol_socp, file)
# #
# #     print(save_path)
# #     # import bioviz
# #     # b = bioviz.Viz(model_path=biorbd_model_path_vision_with_mesh)
# #     # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
# #     # b.exec()
