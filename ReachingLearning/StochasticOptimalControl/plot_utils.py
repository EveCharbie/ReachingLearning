

def set_columns_suptitles(fig, axs):

    column_titles = ["Shoulder muscles", "Elbow muscles", "Biarticular muscles"]

    # Get the position of each column and add titles
    pad = 0.02  # spacing from the top
    for j, col_title in enumerate(column_titles):
        # Calculate x position based on subplot positions
        x_pos = (axs[0, j].get_position().x0 + axs[0, j].get_position().x1) / 2
        y_pos = axs[0, j].get_position().y1 + pad
        fig.text(x_pos, y_pos, col_title, ha="center", va="bottom", fontsize=14, weight="bold")

    return fig, axs