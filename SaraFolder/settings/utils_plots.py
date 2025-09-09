import os

from matplotlib import pyplot as plt

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