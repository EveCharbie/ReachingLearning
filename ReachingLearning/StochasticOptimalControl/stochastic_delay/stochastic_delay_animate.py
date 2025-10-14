from ..stochastic_basic.stochastic_basic_animate import animate_socp_basic


def animate_socp_delay(
    final_time,
    n_shooting,
    q_opt,
    muscles_opt,
):
    animate_socp_basic(
            final_time,
            n_shooting,
            q_opt,
            muscles_opt,
    )
