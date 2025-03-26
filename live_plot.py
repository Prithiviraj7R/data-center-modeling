import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pandas as pd

def live_forecasting_plot():
    # Prepare data for visualization
    k_step = 30
    seq_length = 15

    predictions_df = pd.read_csv('data/predictions_df.csv')
    df = pd.read_csv('data/combined_df.csv')

    full_data = predictions_df['cpu_temp'].values
    prediction_indices = predictions_df.index
    actual_values = predictions_df[[f'actual_cpu_temp_{i+1}' for i in range(k_step)]].values
    predicted_values = predictions_df[[f'predicted_cpu_temp_{i+1}' for i in range(k_step)]].values

    # Create the figure and axis for the live plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Live Forecasting Visualization")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("CPU Temperature")

    # Plot full data
    ax.plot(full_data, 'b--', label="Full Data")

    # Initialize lines for history and predictions
    history_line, = ax.plot([], [], 'g-', label="History Window")
    prediction_line, = ax.plot([], [], 'orange', label="Prediction")
    ax.legend()

    def init():
        """Initialize the plot limits and empty lines."""
        ax.set_xlim(0, len(full_data))
        ax.set_ylim(df['cpu_temp'].min() - 5, df['cpu_temp'].max() + 5)
        history_line.set_data([], [])
        prediction_line.set_data([], [])
        return history_line, prediction_line

    def update(frame):
        """Update the plot for each frame in the animation."""
        if frame >= len(prediction_indices):
            return history_line, prediction_line

        # Define indices for history and prediction
        start_idx = max(0, prediction_indices[frame] - seq_length)
        end_idx = prediction_indices[frame] + k_step

        # Extract data for the current frame
        history_data = full_data[start_idx:prediction_indices[frame]]
        predicted_data = predicted_values[frame]

        # Set data for each line
        history_line.set_data(range(start_idx, prediction_indices[frame]), history_data)
        prediction_line.set_data(range(prediction_indices[frame], end_idx), predicted_data)

        return history_line, prediction_line

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(prediction_indices), init_func=init, blit=True, interval=100)
    plt.show()


if __name__ == "__main__":
    live_forecasting_plot()