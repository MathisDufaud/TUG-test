# Read me TUG-TEST
## Folder structure: 
TUG-TEST
- data_external *(data)*
- data_parkapp *(data)*
- data_synpisa *(data)*
- figures_TUG.pptx *(ppt with figures and updates)*
- old material *(folder with old .py and .csv)*
- readme.md
- requirements.txt *(required libraries)*
- updrs *(updrs data from ParkApp)*
- SaraFolder
  - analysis *(analysis across datasets, individually and together)*
  - results *(results folder)*
    - results all
      - alltests_comments.csv
      - cv_per_test_detailed.csv 
      - cv_results_sample_level.csv
      - cv_results_test_level.csv
      - figures *folder*
      - groundtruths_comparison.csv
      - models *folder*
      - **overviewalltests.txt**
      - resultsall_alllabelling_help_gwalk.txt
      - resultscvfolds_mdl_strongbs_sixax_ml_gwalk.txt
      - resultsholdout_mdl_strongbs_sixax_fold4_ml_gwalk.txt
      - skippedtests_motivation.csv
      - **testssupervised_manualmsstartend_new.csv**
    - results_parkapp
    - results_pisatug
    - results_synergy
  - settings *(utils files, classes.py and running_settings.py)*

## Data overview 
### When incoming new data, what to do: 
1. If part of Pisa or Synergy data: add values to participants.csv and tugs.csv files into the folder data_synpisa
2. Run: analysis_dataquality.py with True at utils_dataquality.observingsingletests() --> output file *testssupervised_manualmsstartend_new.csv*
3. Run: analysis_dataquality.py with True at utils_dataquality.observe_groundtruth(all_tests) --> observing groundtruths
4. Run: analysis_dataquality.py with True at utils_synloaders.overview_total() --> .md output describing statistics of tests

### Timestamp overview
1. Pisa_new: 
   1. df_motion: milliseconds from start
   2. df_orientation: same
2. Pisa: 
   1. milliseconds from start and same
3. Syenrgy new: df motion and orientation have in the msFromStart a string datetime object. To be transformed to datetime and then to int milliseconds
4. Synergy old: Dario must have transformed to milliseconds in his processing pipeline. No need to edit. Already in msFromStart.
5. ParkApp: df_raw: msFromStart in milliseconds from 0. I want df_raw msFromStart to be in milliseconds.

## Running algorithms
1. Run: crossdataset_analysis with True. Choose between method 'labelling' and method 'darioalgo'. Consider when labelling method fails, it fallsback on darioalgo. 
2. 



