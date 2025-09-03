from SaraFolder.settings import utils_ML, utils_functions

if __name__ == "__main__":

    # Loading data
    df = utils_functions.ready_df()

    # Algo pipeline
    utils_ML.ml_pipeline(df)

