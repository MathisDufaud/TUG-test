import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import re

from SaraFolder.settings import running_settings


def plot_training_history(history, title):
    # Extract training & validation metrics
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(loss) + 1)

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    # Plot accuracy
    ax1.plot(epochs, acc, 'o-', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'o-', label='Validation Accuracy')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot loss
    ax2.plot(epochs, loss, 'o-', label='Training Loss')
    ax2.plot(epochs, val_loss, 'o-', label='Validation Loss')
    ax2.set_title('Training vs Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(running_settings.figures_path + os.sep + title, dpi=400)
    plt.show()


def plot_labelling_tug(df_plot, res):
    if res is None:
        print(res)
        # Problem we can't plot, return error warning
        raise ValueError("No phases detected, cannot plot.")
    else:
        (t_start, t_end_stand, t_start_turn, t_end_turn, t_start_turn2, t_end_turn2, t_start_sit, t_end) = res

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_plot["relative_timestamp"], df_plot["sqrt(X²+Y²+Z²)"],
             label="Motion (m/s²)", color="blue", linestyle="-")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Acceleration (m/s²)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax1.axvspan(t_start, t_end, color="orange", alpha=0.2, label="Total duration")
    ax1.axvspan(t_start_turn, t_end_turn, color="limegreen", alpha=0.5, label="First turn")
    ax1.axvspan(t_start_turn2, t_end_turn2, color="darkgreen", alpha=0.5, label="Second turn")
    ax1.axvspan(t_start, t_end_stand, color="red", alpha=0.5, label="First turn")
    ax1.axvspan(t_start_sit, t_end, color="pink", alpha=0.5, label="Second turn")

    ax3 = ax1.twinx()
    ax3.plot(df_plot["relative_timestamp"], df_plot["alpha"], label="Alpha (°)", color="red", linestyle="--")
    ax3.plot(df_plot["relative_timestamp"], df_plot["beta"], label="Beta (°)", color="green", linestyle="-.")
    ax3.plot(df_plot["relative_timestamp"], df_plot["gamma"], label="Gamma (°)", color="purple", linestyle=":")

    plt.xticks(rotation=45)

    ax1.grid()
    ax1.legend(loc="upper left")
    ax3.legend(loc="lower right")

    plt.title("Results on motion and orientation")

    plt.show()


def plot_tugtoverview(df_general):
    """ Figure with subplots
    1) Number of test per participant
    2) Frequency distribution of the test per participant
    3) Boxplot of the duration of the test per participant
    4) DurationGT boxplot per participant
    columns: Participant, Session, samples, duration
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    df_count = df_general.groupby('Participant').size()
    df_count.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Number of tests per participant')
    ax1.set_xlabel('Participant')
    ax1.set_ylabel('Number of tests')
    ax1.grid(axis='y')
    ax2 = fig.add_subplot(2, 2, 2)
    df_general['freq'] = df_general['samples'] / df_general['duration']
    df_general.groupby('Participant')['freq'].plot(kind='hist',ax=ax2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_title('Frequency distribution per participant, per test')
    ax2.grid(axis='y')
    ax2.set_ylim(0, 20)
    ax3 = fig.add_subplot(2, 2, 3)
    df_general.boxplot(column='duration', by='Participant', ax=ax3)
    ax3.set_title('Duration of the test per participant (raw)')
    ax3.set_xlabel('Participant')
    ax3.set_ylabel('Duration (s)')
    ax3.grid(axis='y')
    ax4 = fig.add_subplot(2, 2, 4)
    df_general.boxplot(column='durationGT', by='Participant', ax=ax4)
    ax4.set_title('Duration of the test per participant (GT)')
    ax4.set_xlabel('Participant')
    ax4.set_ylabel('Duration (s)')
    ax4.grid(axis='y')


    plt.tight_layout()
    plt.suptitle('')
    plt.savefig(running_settings.figures_path + os.sep + 'tug_overview.jpg', dpi=400)
    plt.show()

    return None



def plot_icc(icc_s, icctype='ICC2'):
    """
    Plot ICC values with confidence intervals for a specific ICC type across different configurations.

    Parameters:
    -----------
    icc_s : dict
        Dictionary containing ICC results for different configurations
    icctype : str
        Type of ICC to plot (default: 'ICC2')
        Options: 'ICC1', 'ICC2', 'ICC3', 'ICC1k', 'ICC2k', 'ICC3k'
    """

    # Extract data for the specified ICC type
    configurations = []
    icc_values = []
    ci_lower = []
    ci_upper = []
    pvalues = []

    for config_name, df in icc_s.items():
        # Find the row with the specified ICC type
        icc_row = df[df['Type'] == icctype]

        if not icc_row.empty:
            configurations.append(config_name)
            icc_values.append(icc_row['ICC'].iloc[0])
            pvalues.append(icc_row['pval'].iloc[0])

            # Parse confidence interval
            ci_str = str(icc_row['CI95%'].iloc[0])
            # Extract numbers from CI string like '[0.26, 0.78]'
            ci_nums = re.findall(r'[\d.]+', ci_str)
            if len(ci_nums) >= 2:
                ci_lower.append(float(ci_nums[0]))
                ci_upper.append(float(ci_nums[1]))
            else:
                ci_lower.append(icc_row['ICC'].iloc[0])
                ci_upper.append(icc_row['ICC'].iloc[0])

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract numbers from configuration names for x-axis (assuming format like '2_p26_s52')
    x_labels = []
    x_positions = []
    for i, config in enumerate(configurations):
        # Extract the first number from the configuration name
        match = re.match(r'(\d+)', config)
        if match:
            x_labels.append(f"Config {match.group(1)}")
            x_positions.append(int(match.group(1)))
        else:
            x_labels.append(config)
            x_positions.append(i + 1)

    # Sort by x_positions to maintain order
    sorted_indices = np.argsort(x_positions)
    x_positions = [x_positions[i] for i in sorted_indices]
    x_labels = [x_labels[i] for i in sorted_indices]
    icc_values = [icc_values[i] for i in sorted_indices]
    ci_lower = [ci_lower[i] for i in sorted_indices]
    ci_upper = [ci_upper[i] for i in sorted_indices]
    pvalues = [pvalues[i] for i in sorted_indices]

    # Calculate error bars
    yerr_lower = [icc_values[i] - ci_lower[i] for i in range(len(icc_values))]
    yerr_upper = [ci_upper[i] - icc_values[i] for i in range(len(icc_values))]
    yerr = [yerr_lower, yerr_upper]

    # Create color map based on p-values
    colors = []
    for pval in pvalues:
        if pval < 0.001:
            colors.append('darkgreen')
        elif pval < 0.01:
            colors.append('green')
        elif pval < 0.05:
            colors.append('orange')
        else:
            colors.append('red')

    # Plot ICC values with error bars
    bars = ax.bar(x_positions, icc_values,
                  yerr=yerr, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Customize the plot
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('ICC Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{icctype} Values with 95% Confidence Intervals',
                 fontsize=14, fontweight='bold', pad=20)

    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line for different ICC interpretation levels
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Poor (0.5)')
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='Good (0.75)')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Excellent (0.9)')

    # Create legend for p-value colors
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='darkgreen', alpha=0.7, label='p < 0.001'),
        Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='p < 0.01'),
        Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.7, label='p < 0.05'),
        Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='p ≥ 0.05')
    ]

    # Add legends
    legend1 = ax.legend(handles=legend_elements, loc='upper left',
                        title='Significance Level', framealpha=0.9)
    ax.add_artist(legend1)

    # Add text annotations for ICC values
    for i, (pos, val, pval) in enumerate(zip(x_positions, icc_values, pvalues)):
        ax.text(pos, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(pos, val - 0.05, f'p={pval:.2e}' if pval < 0.001 else f'p={pval:.3f}',
                ha='center', va='top', fontsize=8, style='italic')

    # Set y-axis limits
    y_min = min(ci_lower) - 0.1
    y_max = min(max(ci_upper) + 0.15, 1.0)
    ax.set_ylim(y_min, y_max)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

    # Print summary statistics
    print(f"\n{icctype} Summary Statistics:")
    print("=" * 40)
    print(f"Mean ICC: {np.mean(icc_values):.3f}")
    print(f"Std ICC: {np.std(icc_values):.3f}")
    print(f"Min ICC: {np.min(icc_values):.3f}")
    print(f"Max ICC: {np.max(icc_values):.3f}")
    print(f"Configurations with ICC > 0.75: {sum(1 for x in icc_values if x > 0.75)}/{len(icc_values)}")
    print(f"Configurations with ICC > 0.9: {sum(1 for x in icc_values if x > 0.9)}/{len(icc_values)}")

    return fig, ax