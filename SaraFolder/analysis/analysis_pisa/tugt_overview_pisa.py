from SaraFolder.settings import utils_plots
from SaraFolder.settings.utils_parkaapp import utils_parkapp
import matplotlib

from SaraFolder.settings.utils_pisa import utils_pisatugloaders

matplotlib.use('TkAgg')

if __name__ == "__main__":

    tug_data = utils_pisatugloaders.load_json_data()

    all_tests = utils_pisatugloaders.load_pisatests()


    print(1)

