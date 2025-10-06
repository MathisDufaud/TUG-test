data_path = r"C:\Users\ao4518\Desktop\PHD\TUG-test\all_data"
figures_path = r"C:\Users\ao4518\Desktop\PHD\TUG-test\SaraFolder\results_s\figures"
results_path = r"C:\Users\ao4518\Desktop\PHD\TUG-test\SaraFolder\results_s"
models_path = r"C:\Users\ao4518\Desktop\PHD\TUG-test\SaraFolder\results_s\models"

# name_df_processed = "df_processed.pickle" # name of the processed dataframe - LSTM
name_df_processed = "dfprocessed_labelling.pickle" # name of the processed dataframe - LABELLING
res_name = "results_labelling.txt"
tugt_overview = "tugt_overview.txt"

load_existing_model = True
phases_to_consider = 1 # 1: tugt

param_labelling_algo = {
    'thresh_amplitude': 120,     # amplitude
    'thresh_neighbors': 0.3,     # amplitude
    'thresh_turn_duration': 50,  # samples - approx 0.8 seconds
    'window_size': 100,          # samples - approx 1.7 seconds
    'ma_window': 20,             # samples - approx 1/3 seconds
    'k_reverse2': 0.04,
    'min_duration_reverse2':30,  # samples - approx 0.5 seconds
    'k_end2': 0.05,
    'minduration_end2':30,       # samples - approx 0.5 seconds
    'k_all': 0.15,
    'thresh_firstpeak': 1.5,
    'thresh_lastpeak': 1.2,
    'thresh_peak_rotbetagamma': 70,
}