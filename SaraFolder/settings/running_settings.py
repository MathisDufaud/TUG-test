##################### Data
data_parkapp = r"C:\Users\ao4518\Desktop\PHD\TUG-test\data_parkapp"
data_synpisa = r"C:\Users\ao4518\Desktop\PHD\TUG-test\data_synpisa"
test_id_start_parkapp = 0
test_id_start_synergy = 0
test_id_start_pisatug = 0

##################### Results
results_path = r"C:\Users\ao4518\Desktop\PHD\TUG-test\SaraFolder\results"
results_parkapp = results_path + r"\results_parkapp"
results_synergy = results_path + r"\results_synergy"
results_pisatug = results_path + r"\results_pisatug"
results_all = results_path + r"\results_all"

##################### Figures
figures_parkapp = results_parkapp + r"\figures"
figures_synergy = results_synergy + r"\figures"
figures_pisatug = results_pisatug + r"\figures"
figures_all = results_all + r"\figures"

##################### Models
models_parkapp = results_parkapp + r"\models"
models_path = results_all + r"\models"

##################### Variables names
########## Park App Tests
# name_df_processed = "df_processed.pickle" # name of the processed dataframe - LSTM
name_df_processed = "dfprocessed_labelling.pickle" # name of the processed dataframe - LABELLING
res_name = "results_labelling.txt"
tugt_overview_parkapp = "tugt_overview_parkapp.txt"
tugt_overview_pisa = "tugt_overview_pisa.txt"
tugt_overview_synergy = "tugt_overview_synergy.txt"

load_existing_model = False
model_comments = 'first_model_labelling_all_datasets'
phases_to_consider = 1 # 1: tugt
parameters = {
    'resamplingdelta': '16.666667ms',
    'stride': 15,  # samples
}

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

param_opt_labelling_algo = {
    'window_segment_turns': [100, 150, 200],
    'max_alpha_amplitude': [100, 120, 140],
    'max_amplitutde_start_turn_sample':[0.25, 0.3, 0.35],
    'MCSFD':  [20, 30, 40],
}


