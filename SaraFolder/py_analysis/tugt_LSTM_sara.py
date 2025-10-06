from SaraFolder.settings import utils_ML
from SaraFolder.settings.utils_parkaapp import utils_functions

if __name__ == "__main__":

    # Loading data
    df = utils_functions.ready_df()

    # Algo pipeline
    utils_ML.ml_pipeline(df)

