from datetime import datetime

from ReachingLearning import (
    ExampleType,
    run_ocp,
    run_ocp_multimodel,
    run_socp_basic,
    run_socp_delay,
)


RUN_OCP = False
RUN_OCP_MULTIMODEL = False
RUN_SOCP_BASIC = True
RUN_SOCP_DELAY = False
print(RUN_OCP, RUN_OCP_MULTIMODEL, RUN_SOCP_BASIC, RUN_SOCP_DELAY)
print(datetime.now().strftime("%d-%m %H:%M:%S"))

PLOT_FLAG = True
ANIMATE_FLAG = True
example_type = ExampleType.CIRCLE
force_field_magnitude = 0


n_random = 15
n_threads = 7
n_simulations = 30
seed = 0

n_q = 2
dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)
tol = 1e-6

motor_noise_std = 0.05  # taus
wPq_std = 0.00000001
wPqdot_std = 0.00000001


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

# run_ocp_multimodel(
#     final_time=final_time,
#     n_shooting=n_shooting,
#     motor_noise_std=motor_noise_std,
#     example_type=example_type,
#     n_random=n_random,
#     seed=seed,
#     n_threads=n_threads,
#     tol=tol,
#     n_simulations=n_simulations,
#     RUN_OCP_MULTIMODEL=RUN_OCP_MULTIMODEL,
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

# run_socp_delay(
#     final_time=final_time,
#     n_shooting=n_shooting,
#     motor_noise_std=motor_noise_std,
#     wPq_std=wPq_std,
#     wPqdot_std=wPqdot_std,
#     force_field_magnitude=force_field_magnitude,
#     example_type=example_type,
#     n_random=n_random,
#     seed=seed,
#     n_threads=n_threads,
#     tol=tol,
#     n_simulations=n_simulations,
#     RUN_SOCP_BASIC=RUN_SOCP_BASIC,
#     PLOT_FLAG=PLOT_FLAG,
#     ANIMATE_FLAG=ANIMATE_FLAG,
# )
