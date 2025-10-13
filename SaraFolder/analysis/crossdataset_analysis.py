from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation
from SaraFolder.settings.utils_parkaapp import utils_parkapp
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    print(1)
    all_tests_parkapp = utils_parkapp.load_all_tests(dataset_id ='parkapp')
    all_tests_pisa = utils_parkapp.load_all_tests(dataset_id = 'pisa')
    all_tests_synergy = utils_parkapp.load_all_tests(dataset_id = 'synergy')

