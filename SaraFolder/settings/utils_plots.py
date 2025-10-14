import os
import matplotlib
matplotlib.use('TkAgg')
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
    plt.tight_layout()
    plt.savefig(running_settings.figures_all + os.sep + 'results_all.jpg', dpi=400)

def plot_tugtoverview(df_general, image_path):
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
    df_general.boxplot(column='durationGTm', by='Participant', ax=ax4)
    ax4.set_title('Duration of the test per participant (GT)')
    ax4.set_xlabel('Participant')
    ax4.set_ylabel('Duration (s)')
    ax4.grid(axis='y')


    plt.tight_layout()
    plt.suptitle('')
    plt.savefig(image_path, dpi=400)
    plt.show()

    return None

def plot_icc(icc_s, icctype='ICC2'):

    # Extract data for the specified ICC type
    configurations = []
    icc_values = []
    ci_lower = []
    ci_upper = []
    pvalues = []
    num_tests = []
    num_participants = []

    for config_name, df in icc_s.items():
        # Find the row with the specified ICC type
        icc_row = df[df['Type'] == icctype]

        if not icc_row.empty:
            configurations.append(config_name)
            icc_values.append(icc_row['ICC'].iloc[0])
            pvalues.append(icc_row['pval'].iloc[0])

            # Parse configuration name to extract numbers
            # Expected format: 'numTests_pNumParticipants_sNumSamples'
            # Example: '2_p26_s52' means 2 tests, 26 participants, 52 samples
            match = re.match(r'(\d+)_p(\d+)', config_name)
            if match:
                num_tests.append(int(match.group(1)))
                num_participants.append(int(match.group(2)))
            else:
                # Fallback if pattern doesn't match
                num_tests.append(len(configurations))
                num_participants.append(len(configurations))

            # Parse confidence interval
            ci_str = str(icc_row['CI95%'].iloc[0])
            ci_nums = re.findall(r'[\d.]+', ci_str)
            if len(ci_nums) >= 2:
                ci_lower.append(float(ci_nums[0]))
                ci_upper.append(float(ci_nums[1]))
            else:
                ci_lower.append(icc_row['ICC'].iloc[0])
                ci_upper.append(icc_row['ICC'].iloc[0])

    # Create figure with two subplots side by side
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    # =========================================================================
    # SUBPLOT 1: ICC vs Number of Tests (Original Plot)
    # =========================================================================

    # Sort by number of tests
    sorted_indices_tests = np.argsort(num_tests)
    x_positions_tests = [num_tests[i] for i in sorted_indices_tests]
    icc_sorted_tests = [icc_values[i] for i in sorted_indices_tests]
    ci_lower_sorted_tests = [ci_lower[i] for i in sorted_indices_tests]
    ci_upper_sorted_tests = [ci_upper[i] for i in sorted_indices_tests]
    pvalues_sorted_tests = [pvalues[i] for i in sorted_indices_tests]

    # Calculate error bars
    yerr_lower_tests = [icc_sorted_tests[i] - ci_lower_sorted_tests[i]
                        for i in range(len(icc_sorted_tests))]
    yerr_upper_tests = [ci_upper_sorted_tests[i] - icc_sorted_tests[i]
                        for i in range(len(icc_sorted_tests))]
    yerr_tests = [yerr_lower_tests, yerr_upper_tests]

    # Create color map based on p-values
    colors_tests = []
    for pval in pvalues_sorted_tests:
        if pval < 0.001:
            colors_tests.append('darkgreen')
        elif pval < 0.01:
            colors_tests.append('green')
        elif pval < 0.05:
            colors_tests.append('orange')
        else:
            colors_tests.append('red')

    # # Plot ICC values with error bars
    # ax1.bar(x_positions_tests, icc_sorted_tests, yerr=yerr_tests,
    #         capsize=5, color=colors_tests, alpha=0.7, edgecolor='black', linewidth=1)
    #
    # # Customize subplot 1
    # ax1.set_xlabel('Number of Tests', fontsize=12, fontweight='bold')
    # ax1.set_ylabel('ICC Value', fontsize=12, fontweight='bold')
    # ax1.set_title(f'{icctype} vs Number of Tests\n95% Confidence Intervals',
    #               fontsize=14, fontweight='bold', pad=20)
    # ax1.grid(True, alpha=0.3, axis='y')
    #
    # # Add reference lines
    # ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Poor (0.5)')
    # ax1.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='Good (0.75)')
    # ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Excellent (0.9)')
    #
    # # Add legend for p-values
    # legend_elements = [
    #     Rectangle((0, 0), 1, 1, facecolor='darkgreen', alpha=0.7, label='p < 0.001'),
    #     Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='p < 0.01'),
    #     Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.7, label='p < 0.05'),
    #     Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='p ≥ 0.05')
    # ]
    # legend1 = ax1.legend(handles=legend_elements, loc='upper left',
    #                      title='Significance Level', framealpha=0.9)
    #
    # # Add value annotations
    # for i, (pos, val) in enumerate(zip(x_positions_tests, icc_sorted_tests)):
    #     ax1.text(pos, val + 0.02, f'{val:.3f}',
    #              ha='center', va='bottom', fontsize=9, fontweight='bold')
    #
    # Set y-axis limits
    y_min = max(0, min(ci_lower) - 0.1)
    y_max = min(max(ci_upper_sorted_tests) + 0.15, 1.0)

    # =========================================================================
    # SUBPLOT 2: ICC vs Number of Participants (colored by # tests)
    # =========================================================================

    # Group data by number of tests for different colors
    unique_tests = sorted(set(num_tests))

    # Create color palette for different test numbers
    colormap = plt.cm.get_cmap('Set2', len(unique_tests))
    test_colors = {test: colormap(i) for i, test in enumerate(unique_tests)}

    # Sort by number of participants
    sorted_indices_parts = np.argsort(num_participants)
    x_positions_parts = [num_participants[i] for i in sorted_indices_parts]
    icc_sorted_parts = [icc_values[i] for i in sorted_indices_parts]
    ci_lower_sorted_parts = [ci_lower[i] for i in sorted_indices_parts]
    ci_upper_sorted_parts = [ci_upper[i] for i in sorted_indices_parts]
    tests_sorted = [num_tests[i] for i in sorted_indices_parts]

    # Plot each point with error bars
    for i in range(len(x_positions_parts)):
        x = x_positions_parts[i]
        y = icc_sorted_parts[i]
        yerr_low = y - ci_lower_sorted_parts[i]
        yerr_up = ci_upper_sorted_parts[i] - y
        n_test = tests_sorted[i]
        color = test_colors[n_test]

        # Plot point with error bar
        ax2.errorbar(x, y, yerr=[[yerr_low], [yerr_up]],
                     fmt='o', markersize=10, color=color,
                     capsize=5, capthick=2, alpha=0.8,
                     label=f'{n_test} tests' if i == 0 or tests_sorted[i] != tests_sorted[i - 1] else "")

    # Connect points with same number of tests
    for test_num in unique_tests:
        # Get all points with this test number
        indices = [i for i, t in enumerate(tests_sorted) if t == test_num]
        if len(indices) > 1:
            x_vals = [x_positions_parts[i] for i in indices]
            y_vals = [icc_sorted_parts[i] for i in indices]
            # Sort by x for proper line connection
            sorted_pairs = sorted(zip(x_vals, y_vals))
            x_vals, y_vals = zip(*sorted_pairs)
            ax2.plot(x_vals, y_vals, '--', color=test_colors[test_num],
                     alpha=0.4, linewidth=1.5)

    # Customize subplot 2
    ax2.set_xlabel('Number of Participants', fontsize=12)
    ax2.set_ylabel('ICC Value', fontsize=12)
    ax2.set_title(f'{icctype} vs Number of Participants\n95% Confidence Intervals',
                  fontsize=12, pad=20)
    ax2.grid(True, alpha=0.3)

    # Add reference lines
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)

    # Create legend for number of tests
    handles, labels = ax2.get_legend_handles_labels()
    # Remove duplicates while preserving order
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(),
               loc='best', title='Number of Tests', framealpha=0.9)

    # Add value annotations
    for i, (x, y) in enumerate(zip(x_positions_parts, icc_sorted_parts)):
        ax2.text(x, y + 0.02, f'{y:.3f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Set y-axis limits
    ax2.set_ylim(y_min, y_max)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(running_settings.figures_parkapp + os.sep + 'icc.jpg', dpi=400)
    plt.show()

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================

    print(f"\n{icctype} Summary Statistics:")
    print("=" * 60)
    print(f"Mean ICC: {np.mean(icc_values):.3f} ± {np.std(icc_values):.3f}")
    print(f"Range: [{np.min(icc_values):.3f}, {np.max(icc_values):.3f}]")
    print(f"\nNumber of Tests Range: {min(num_tests)} - {max(num_tests)}")
    print(f"Number of Participants Range: {min(num_participants)} - {max(num_participants)}")
    print("\nConfigurations tested:")
    for config, icc, pval, n_test, n_part in zip(configurations, icc_values,
                                                 pvalues, num_tests, num_participants):
        print(f"  {config}: ICC={icc:.3f}, p={pval:.4f}, "
              f"Tests={n_test}, Participants={n_part}")
